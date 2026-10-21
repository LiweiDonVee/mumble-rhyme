"""Pinned Qwen3.5 adapter; no heavyweight imports until inference/download."""

import json
import sys
import threading
from pathlib import Path

MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MODEL_SPECS = {
    "9b": (MODEL_ID, MODEL_REVISION),
    "4b": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
}
MAX_INPUT_TOKENS = 2048


def scored_token_positions(offsets, boundary):
    return [i for i, (start, end) in enumerate(offsets) if end > boundary and end > start]


def assemble_completion(left, candidate, right):
    prefix = left + (" " if left and not left[-1].isspace() else "")
    suffix = (" " if right and not right[0].isspace() and right[0] not in ",.!?;:)]}" else "") + right
    return prefix + candidate + suffix, len(prefix)


class LocalModel:
    def __init__(self, quantization="nf4", model="9b"):
        if quantization not in {"nf4", "none"}:
            raise ValueError("quantization must be nf4 or none")
        self.quantization = quantization
        if model not in MODEL_SPECS:
            raise ValueError("model must be 9b or 4b")
        self.model_id, self.revision = MODEL_SPECS[model]
        self.model = None
        self.tokenizer = None
        self.lock = threading.RLock()

    def status(self):
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "tokenizer_revision": self.revision,
            "quantization": self.quantization,
            "loaded": self.model is not None,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "backend": "transformers",
            "automatic_model_substitution": False,
        }

    def download(self):
        from huggingface_hub import snapshot_download

        return snapshot_download(
            self.model_id,
            revision=self.revision,
            allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors", "LICENSE"],
            max_workers=3,
        )

    def load(self):
        with self.lock:
            if self.model is not None:
                return
            import torch
            from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

            if self.quantization == "nf4" and not torch.cuda.is_available():
                raise RuntimeError(
                    "NF4 requires CUDA here. Install a compatible CUDA PyTorch, or select --quantization none."
                )
            kwargs = {
                "revision": self.revision,
                "device_map": "auto",
                "dtype": torch.bfloat16,
                "trust_remote_code": False,
                "attn_implementation": "sdpa",
            }
            if self.quantization == "nf4":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                kwargs["device_map"] = {"": 0}
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, revision=self.revision, trust_remote_code=False
            )
            model_path = self.model_id
            if sys.platform == "win32":
                from .checkpoint import reshard_checkpoint

                snapshot = Path(self.download())
                model_path = str(reshard_checkpoint(
                    snapshot, snapshot.parent.parent / "mumble-rhyme-shards" / self.revision
                ))
            model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, **kwargs).eval()
            if len(tokenizer) > model.get_input_embeddings().num_embeddings:
                raise RuntimeError("Tokenizer vocabulary exceeds pinned model embeddings")
            self.tokenizer, self.model = tokenizer, model

    def _prompt(self, intent=""):
        messages = [
            {
                "role": "system",
                "content": "You are an English songwriter. Write natural original English lyrics. "
                "Return only the lyric text requested, without explanation.",
            },
            {
                "role": "user",
                "content": "Continue this lyric naturally."
                + (" Meaning and mood: " + intent if intent else ""),
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def score(self, candidates, left="", right="", intent=""):
        self.load()
        import torch

        scores = []
        with self.lock, torch.inference_mode():
            for candidate in candidates:
                full, boundary = assemble_completion(left, candidate, right)
                prompt = self._prompt(intent)
                text = prompt + full
                boundary += len(prompt)
                encoded = self.tokenizer(
                    text, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=False
                )
                offsets = encoded.pop("offset_mapping")[0].tolist()
                if encoded["input_ids"].shape[1] > MAX_INPUT_TOKENS:
                    raise ValueError(
                        f"Context and completion exceed {MAX_INPUT_TOKENS} tokens; shorten the input"
                    )
                positions = [i for i in scored_token_positions(offsets, boundary) if i > 0]
                if not positions:
                    raise ValueError("Completion has no scoreable tokens")
                inputs = {k: v.to(self.model.device) for k, v in encoded.items()}
                # Materialize logits only for prediction positions: large vocabulary otherwise wastes VRAM.
                predict_at = torch.tensor([i - 1 for i in positions], device=self.model.device)
                output = self.model(**inputs, use_cache=False, logits_to_keep=predict_at)
                logits = output.logits[0].float()
                targets = inputs["input_ids"][0, positions]
                logprobs = logits.log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)
                scores.append(
                    {
                        "mean_logprob": logprobs.mean().item(),
                        "sum_logprob": logprobs.sum().item(),
                        "scored_tokens": len(positions),
                    }
                )
        return scores

    def generate(self, left, right, intent, target, count=8):
        if not 1 <= count <= 12:
            raise ValueError("count must be between 1 and 12")
        self.load()
        import torch

        request = {
            "left_context": left,
            "right_context": right,
            "meaning_or_mood": intent,
            "rhyme_reference": target,
            "number_of_options": count,
        }
        messages = [
            {
                "role": "system",
                "content": "Help a nonnative speaker write original English song lyrics. "
                "The meaning_or_mood is the primary brief: preserve its concrete actions, objects, "
                "and emotional conflict in every option. If the brief is Chinese, express that meaning "
                "naturally in English. Avoid replacing a specific situation with generic mood words. "
                "The user provides context around a gap and possibly a Chinese meaning. "
                "Return a JSON array of distinct English strings to put into the gap. "
                "When context is empty, each string should be one complete lyric line. "
                "rhyme_reference describes a SOUND, not the topic of the lyric. "
                "Aim for each option to end in a rhyme or near rhyme with rhyme_reference. "
                "Keep the requested meaning; vary the ending words instead of repeating the reference. "
                "Offer varied natural wording, including near rhymes. No commentary, no Markdown. "
                "Do not repeat the provided left or right context. "
                'Your entire response MUST be a valid JSON array of strings: ["lyric option", "another option"]. '
                "Do not return unquoted plain lines or explanations.",
            },
            {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        with self.lock, torch.inference_mode():
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
                self.model.device
            )
            if inputs.input_ids.shape[1] > MAX_INPUT_TOKENS:
                raise ValueError(f"Proposal prompt exceeds {MAX_INPUT_TOKENS} tokens")
            output = self.model.generate(
                **inputs,
                max_new_tokens=384,
                do_sample=True,
                temperature=0.85,
                top_p=0.95,
                top_k=30,
                repetition_penalty=1.05,
            )
            text = self.tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        candidates = parse_proposals(text)
        if not candidates:
            raise RuntimeError(f"Model returned no usable candidates. Raw output: {text[:500]}")
        return candidates[:count], text


def parse_proposals(text):
    # Parse the actual JSON array, tolerating code fences/prose without inventing options.
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "[":
            continue
        try:
            data, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return list(dict.fromkeys(x.strip() for x in data if x.strip()))
    return []
