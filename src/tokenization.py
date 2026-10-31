
from functools import lru_cache

TOKENIZER_NAME = "openai/clip-vit-base-patch32"


@lru_cache(maxsize=4)
def get_tokenizer(cache_dir=None, local_files_only=False):
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        TOKENIZER_NAME, cache_dir=cache_dir, local_files_only=local_files_only)
    if len(tokenizer) != 49408 or tokenizer.eos_token_id != 49407:
        raise ValueError("Tokenizer does not match the custom encoder vocabulary")
    return tokenizer


def tokenize(captions, context_length=77, cache_dir=None):
    return get_tokenizer(cache_dir)(captions, padding="max_length", truncation=True,
                                     max_length=context_length, return_tensors="pt")["input_ids"]
