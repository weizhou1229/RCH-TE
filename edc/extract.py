from typing import List
import os
from pathlib import Path
import edc.utils.llm_utils as llm_utils
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging


logger = logging.getLogger(__name__)


class Extractor:
    # The class to handle the first stage: Open Information Extraction
    def __init__(self, model: AutoModelForCausalLM = None, tokenizer: AutoTokenizer = None, openai_model=None) -> None:
        assert openai_model is not None or (model is not None and tokenizer is not None)
        self.model = model
        self.tokenizer = tokenizer
        self.openai_model = openai_model

    def extract(
        self,
        input_text_str: str,
        few_shot_examples_str: str,
        prompt_template_str: str,
        entities_hint: str = None,
        relations_hint: str = None,
    ) -> List[List[str]]:
        assert (entities_hint is None and relations_hint is None) or (
            relations_hint is not None and relations_hint is not None
        )

        filled_prompt = prompt_template_str.format_map(
            {
                "few_shot_examples": few_shot_examples_str,
                "input_text": input_text_str,
                "entities_hint": entities_hint,
                "relations_hint": relations_hint,
            }
        )

        messages = [{"role": "user", "content": filled_prompt}]

        if self.openai_model is None:
            # llm_utils.generate_completion_transformers([messages], self.model, self.tokenizer, device=self.device)
            completion = llm_utils.generate_completion_transformers(
                messages, self.model, self.tokenizer, answer_prepend="Triplets: "
            )
        else:
            completion = llm_utils.openai_chat_completion(self.openai_model, None, messages)
        extracted_triplets_list = llm_utils.parse_raw_triplets(completion)
        logger.warning(f"OIE raw completion:\n{completion}\n")
        logger.warning(f"OIE parsed triplets: {extracted_triplets_list}")

        if len(extracted_triplets_list) == 0 and input_text_str.strip() != "":
            logger.warning(
                "OIE returned no parsable triples on the first attempt. Retrying with a stricter prompt."
            )
            logger.warning(f"OIE input text for retry: {input_text_str}")
            retry_prompt = (
                filled_prompt
                + "\nYour previous answer did not contain usable triples. "
                + "Return a valid Python list of triples now. "
                + "If the text states any factual relation, extract it. "
                + "Output only the list."
            )
            retry_messages = [{"role": "user", "content": retry_prompt}]
            if self.openai_model is None:
                completion = llm_utils.generate_completion_transformers(
                    retry_messages, self.model, self.tokenizer, answer_prepend="Triplets: "
                )
            else:
                completion = llm_utils.openai_chat_completion(self.openai_model, None, retry_messages)
            extracted_triplets_list = llm_utils.parse_raw_triplets(completion)
            logger.warning(f"OIE retry raw completion:\n{completion}\n")
            logger.warning(f"OIE retry parsed triplets: {extracted_triplets_list}")
            if len(extracted_triplets_list) == 0:
                logger.warning("OIE retry still returned no parsable triples.")
        return extracted_triplets_list
