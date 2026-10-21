"""Validate the adapter API with pinned real tokenizer + random tiny Qwen3.5.

This does NOT validate 9B weights, lyric quality, NF4 or CUDA. It runs on CPU.
"""

import json

import torch
from transformers import AutoTokenizer, Qwen3_5Config, Qwen3_5ForConditionalGeneration

from mumble_rhyme.model import MODEL_ID, MODEL_REVISION, LocalModel


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    config = Qwen3_5Config(
        text_config={
            "vocab_size": 248320,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "rope_parameters": {
                "rope_type": "default",
                "mrope_section": [2, 3, 3],
                "mrope_interleaved": True,
            },
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "depth": 1,
            "num_heads": 2,
            "out_hidden_size": 64,
            "num_position_embeddings": 16,
        },
    )
    model = Qwen3_5ForConditionalGeneration(config).eval()
    adapter = LocalModel(quantization="none")
    adapter.model, adapter.tokenizer = model, tokenizer
    scores = adapter.score(["emotionally unavailable"], left="I feel", right=", tonight.")
    prompt = adapter._prompt()
    text = prompt + "I feel emotionally unavailable, tonight."
    boundary = len(prompt + "I feel ")
    encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoded.pop("offset_mapping")[0].tolist()
    selected = [i for i, (_, end) in enumerate(offsets) if i > 0 and end > boundary]
    with torch.inference_mode():
        logits = model(**encoded, use_cache=False).logits[0, :-1].float()
        logprobs = logits.log_softmax(-1).gather(-1, encoded.input_ids[0, 1:, None]).squeeze(-1)
    expected = logprobs[[i - 1 for i in selected]].mean().item()
    assert abs(scores[0]["mean_logprob"] - expected) < 1e-5, (scores, expected)
    assert scores[0]["scored_tokens"] == len(selected)
    inputs = tokenizer(prompt + "I feel", return_tensors="pt", add_special_tokens=False)
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=2, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    assert generated.shape[1] == inputs.input_ids.shape[1] + 2
    print(
        json.dumps(
            {
                "adapter_score_matches_full_logits": True,
                "two_token_cached_generation": True,
                "tokenizer_revision": MODEL_REVISION,
                "tokenizer_vocabulary": len(tokenizer),
                "test_weights": "random tiny Qwen3.5, NOT the 9B model",
                "scores": scores,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
