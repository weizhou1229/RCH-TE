import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class ShardedSentenceEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.prompts = {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto",
        )
        self.model.eval()
        self.input_device = self._infer_input_device()

    def _infer_input_device(self):
        if hasattr(self.model, "hf_device_map"):
            for device in self.model.hf_device_map.values():
                if isinstance(device, int):
                    return torch.device(f"cuda:{device}")
                if isinstance(device, str) and device not in {"cpu", "disk"}:
                    return torch.device(device)
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _prepare_text(self, text: str, prompt_name=None, prompt=None):
        if prompt is not None:
            return f"{prompt}{text}"
        if prompt_name is not None and prompt_name in self.prompts:
            return f"{self.prompts[prompt_name]}{text}"
        return text

    def _last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def encode(self, texts, prompt_name=None, prompt=None):
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        prepared_texts = [self._prepare_text(text, prompt_name=prompt_name, prompt=prompt) for text in texts]
        max_length = min(getattr(self.tokenizer, "model_max_length", 4096), 4096)
        batch_dict = self.tokenizer(
            prepared_texts,
            max_length=max_length - 1,
            return_attention_mask=False,
            padding=False,
            truncation=True,
        )
        batch_dict["input_ids"] = [input_ids + [self.tokenizer.eos_token_id] for input_ids in batch_dict["input_ids"]]
        batch_dict = self.tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors="pt")
        batch_dict = {k: v.to(self.input_device) for k, v in batch_dict.items()}

        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = self._last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)

        embeddings = embeddings.detach().cpu().numpy()
        if single_input:
            return embeddings[0]
        return embeddings
