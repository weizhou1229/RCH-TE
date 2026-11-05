import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import ast
from sentence_transformers import SentenceTransformer
from typing import List
import gc
import torch
import logging
from openai import OpenAI
import json

logger = logging.getLogger(__name__)
MODEL_REGISTRY = {}


def _expand_env_var(value):
    if not isinstance(value, str):
        return value
    return os.path.expandvars(value)


def set_model_registry(config_path: str = None):
    global MODEL_REGISTRY
    MODEL_REGISTRY = {}

    if config_path is None or config_path == "":
        return
    if not os.path.exists(config_path):
        logger.warning(f"Model config file {config_path} does not exist. Falling back to env/default settings.")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "models" in payload:
        payload = payload["models"]

    if not isinstance(payload, dict):
        raise ValueError("Model config file must be a JSON object or contain a top-level 'models' object.")

    MODEL_REGISTRY = payload
    logger.info(f"Loaded {len(MODEL_REGISTRY)} model configuration(s) from {config_path}")


def get_model_config(model_name: str):
    config = MODEL_REGISTRY.get(model_name)
    if not isinstance(config, dict):
        return None

    normalized = dict(config)
    for key in ["model", "api_key", "base_url", "backend", "type"]:
        if key in normalized:
            normalized[key] = _expand_env_var(normalized[key])

    api_key_env = normalized.get("api_key_env")
    if api_key_env and not normalized.get("api_key"):
        normalized["api_key"] = os.environ.get(api_key_env)

    base_url_env = normalized.get("base_url_env")
    if base_url_env and not normalized.get("base_url"):
        normalized["base_url"] = os.environ.get(base_url_env)

    return normalized


def resolve_model_name(model_name: str):
    config = get_model_config(model_name)
    if config is None:
        return model_name
    return config.get("model", model_name)


def free_model(model: AutoModelForCausalLM = None, tokenizer: AutoTokenizer = None):
    try:
        if model is not None and hasattr(model, "cpu"):
            model.cpu()
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(e)


def get_embedding_e5mistral(model, tokenizer, sentence, task=None):
    model.eval()
    device = model.device

    if task != None:
        # It's a query to be embed
        sentence = get_detailed_instruct(task, sentence)

    sentence = [sentence]

    max_length = 4096
    # Tokenize the input texts
    batch_dict = tokenizer(
        sentence, max_length=max_length - 1, return_attention_mask=False, padding=False, truncation=True
    )
    # append eos_token_id to every input_ids
    batch_dict["input_ids"] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict["input_ids"]]
    batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors="pt")

    batch_dict.to(device)

    embeddings = model(**batch_dict).detach().cpu()

    assert len(embeddings) == 1

    return embeddings[0]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


def get_embedding_sts(model: SentenceTransformer, text: str, prompt_name=None, prompt=None):
    embedding = model.encode(text, prompt_name=prompt_name, prompt=prompt)
    return embedding


def parse_raw_entities(raw_entities: str):
    parsed_entities = []
    left_bracket_idx = raw_entities.index("[")
    right_bracket_idx = raw_entities.index("]")
    try:
        parsed_entities = ast.literal_eval(raw_entities[left_bracket_idx : right_bracket_idx + 1])
    except Exception as e:
        pass
    logging.debug(f"Entities {raw_entities} parsed as {parsed_entities}")
    return parsed_entities


def parse_raw_triplets(raw_triplets: str):
    # Look for enclosing brackets
    unmatched_left_bracket_indices = []
    matched_bracket_pairs = []

    collected_triples = []
    for c_idx, c in enumerate(raw_triplets):
        if c == "[":
            unmatched_left_bracket_indices.append(c_idx)
        if c == "]":
            if len(unmatched_left_bracket_indices) == 0:
                continue
            # Found a right bracket, match to the last found left bracket
            matched_left_bracket_idx = unmatched_left_bracket_indices.pop()
            matched_bracket_pairs.append((matched_left_bracket_idx, c_idx))
    for l, r in matched_bracket_pairs:
        bracketed_str = raw_triplets[l : r + 1]
        try:
            parsed_triple = ast.literal_eval(bracketed_str)
            if len(parsed_triple) == 3 and all([isinstance(t, str) for t in parsed_triple]):
                if all([e != "" and e != "_" for e in parsed_triple]):
                    collected_triples.append(parsed_triple)
            elif not all([type(x) == type(parsed_triple[0]) for x in parsed_triple]):
                for e_idx, e in enumerate(parsed_triple):
                    if isinstance(e, list):
                        parsed_triple[e_idx] = ", ".join(e)
                collected_triples.append(parsed_triple)
        except Exception as e:
            pass
    logger.debug(f"Triplets {raw_triplets} parsed as {collected_triples}")
    return collected_triples


def parse_relation_definition(raw_definitions: str):
    descriptions = raw_definitions.split("\n")
    relation_definition_dict = {}

    for description in descriptions:
        if ":" not in description:
            continue
        index_of_colon = description.index(":")
        relation = description[:index_of_colon].strip()

        relation_description = description[index_of_colon + 1 :].strip()

        if relation == "Answer":
            continue

        relation_definition_dict[relation] = relation_description
    logger.debug(f"Relation Definitions {raw_definitions} parsed as {relation_definition_dict}")
    return relation_definition_dict


def is_model_openai(model_name):
    registered_config = get_model_config(model_name)
    if registered_config is not None:
        backend = (registered_config.get("backend") or registered_config.get("type") or "").lower()
        return backend in {"openai", "openai_compatible", "api"}

    model_name = model_name.lower()
    return "gpt" in model_name or "deepseek" in model_name


def generate_completion_transformers(
    input: list,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_token=256,
    answer_prepend="",
):
    device = model.device
    tokenizer.pad_token = tokenizer.eos_token

    messages = tokenizer.apply_chat_template(input, add_generation_prompt=True, tokenize=False) + answer_prepend

    model_inputs = tokenizer(messages, return_tensors="pt", padding=True, add_special_tokens=False).to(device)

    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=max_new_token,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )

    generation = model.generate(**model_inputs, generation_config=generation_config)
    sequences = generation["sequences"]
    generated_ids = sequences[:, model_inputs["input_ids"].shape[1] :]
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    logging.debug(f"Prompt:\n {messages}\n Result: {generated_texts}")
    return generated_texts


def openai_chat_completion(model, system_prompt, history, temperature=0, max_tokens=512):
    model_config = get_model_config(model)
    if model_config is None:
        raise ValueError(
            f"Model '{model}' is not defined in model_api_config.json. "
            "Add a config entry before using this model alias."
        )

    resolved_model_name = model_config.get("model", model)
    api_key = model_config.get("api_key")
    base_url = model_config.get("base_url")

    if not api_key:
        raise ValueError(
            f"Missing api_key for model '{model}' in model_api_config.json. "
            "Set 'api_key' directly or provide 'api_key_env' with a valid environment variable."
        )

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    response = None
    if system_prompt is not None:
        messages = [{"role": "system", "content": system_prompt}] + history
    else:
        messages = history
    while response is None:
        try:
            response = client.chat.completions.create(
                model=resolved_model_name, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
        except Exception as e:
            logger.warning(f"OpenAI-compatible request failed for model '{model}' ({resolved_model_name}): {e}")
            status_code = getattr(e, "status_code", None)
            if status_code is not None and 400 <= status_code < 500 and status_code != 429:
                raise
            time.sleep(5)
    logging.debug(
        f"Model alias: {model}\nResolved model: {resolved_model_name}\nPrompt:\n {messages}\n Result: {response.choices[0].message.content}"
    )
    return response.choices[0].message.content
