"""Pinned Qwen3.5 adapter; no heavyweight imports until inference/download."""

import gc
import json
import sys
import threading
import traceback
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MODEL_SPECS = {
    "9b": (MODEL_ID, MODEL_REVISION),
    "4b": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
}
MAX_INPUT_TOKENS = 2048
_RUNTIME_LOCK = threading.RLock()


def _inference_operation(method):
    @wraps(method)
    def run(self, *args, **kwargs):
        # Keep allocation-owning method frames outside the cleanup handler:
        # they must have unwound before their traceback locals can be cleared.
        with self._inference():
            return method(self, *args, **kwargs)

    return run


def _clear_error_frames(error):
    pending, visited = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        traceback.clear_frames(current.__traceback__)
        pending.extend(e for e in (current.__cause__, current.__context__) if e is not None)


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
        self.lock = _RUNTIME_LOCK
        self.last_resource_error = None
        self.loading_warmup = []

    def status(self):
        from . import __version__

        return {
            "package_version": __version__,
            "model_id": self.model_id,
            "revision": self.revision,
            "tokenizer_revision": self.revision,
            "quantization": self.quantization,
            "loaded": self.model is not None,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "backend": "transformers",
            "automatic_model_substitution": False,
            "last_resource_error": self.last_resource_error,
            "loading_warmup": self.loading_warmup,
        }

    def download(self):
        from huggingface_hub import snapshot_download

        return snapshot_download(
            self.model_id,
            revision=self.revision,
            allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors", "LICENSE"],
            max_workers=3,
        )

    def unload(self, *, release_runtime_cache=False):
        """Release this instance's weights and unused CUDA cache under its lock.

        CUDA runtime/context memory can remain until process exit. References
        held by external Python callers are outside this instance's ownership.
        release_runtime_cache=True also clears process-wide cuBLAS workspaces;
        callers must stop external CUDA activity first. Project adapters share
        a lock, but unrelated torch code cannot be synchronized by that lock.
        """
        with self.lock:
            self.model, self.tokenizer = None, None
            gc.collect()
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_initialized():
                # PyTorch 2.8's cached cuBLAS workspace can keep an otherwise
                # free multi-GB allocator segment alive after model teardown.
                clear_workspaces = getattr(getattr(torch, "_C", None), "_cuda_clearCublasWorkspaces", None)
                if release_runtime_cache and clear_workspaces is not None:
                    torch.cuda.synchronize()
                    clear_workspaces()
                torch.cuda.empty_cache()

    def _prepare_cuda(self):
        """Initialize this thread's cuBLAS/BF16 path before allocating model weights."""
        import torch

        with torch.inference_mode():
            torch.cuda.current_blas_handle()
            probe = torch.ones((16, 16), dtype=torch.bfloat16, device="cuda:0")
            output = torch.nn.functional.linear(probe, probe)
            if output[0, 0].item() != 16:
                raise RuntimeError("CUDA BF16 runtime initialization failed")
            del probe, output
        torch.cuda.synchronize(0)
        torch.cuda.empty_cache()

    @contextmanager
    def _inference(self):
        import torch

        with self.lock, torch.inference_mode():
            try:
                yield
            except (RuntimeError, MemoryError) as error:
                message = str(error)
                resource_error = isinstance(error, MemoryError) or any(
                    marker in message.lower() for marker in (
                        "cuda out of memory", "cuda error: out of memory",
                        "cublas_status_alloc_failed", "cudnn_status_alloc_failed",
                    )
                )
                if not resource_error:
                    raise
                self.last_resource_error = message or type(error).__name__
                _clear_error_frames(error)
                try:
                    self.unload()
                except RuntimeError:
                    # Preserve the primary CUDA failure if its context cannot be reused.
                    self.model, self.tokenizer = None, None
                raise RuntimeError(
                    f"{self.model_id}: {self.last_resource_error}. Model weights released after the failed "
                    "operation. Free system/GPU memory before retrying; restart the server if CUDA remains "
                    "unavailable. Choose --model 4b explicitly for the smaller model."
                ) from None
            else:
                self.last_resource_error = None

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
            model = None
            try:
                if self.quantization == "nf4":
                    self._prepare_cuda()
                if sys.platform == "win32":
                    from transformers import GenerationConfig, Qwen3_5Config

                    from .checkpoint import full_tensor_state_dict, reshard_checkpoint
                    from .loading import bounded_allocator_warmup

                    snapshot = Path(self.download())
                    directory = reshard_checkpoint(
                        snapshot, snapshot.parent.parent / "mumble-rhyme-shards" / self.revision
                    )
                    kwargs["config"] = Qwen3_5Config.from_pretrained(snapshot, trust_remote_code=False)
                    if (snapshot / "generation_config.json").is_file():
                        kwargs["generation_config"] = GenerationConfig.from_pretrained(snapshot)
                    device = "cuda:0" if self.quantization == "nf4" else "cpu"
                    with (full_tensor_state_dict(directory, device=device) as state,
                          bounded_allocator_warmup() as warmup):
                        self.loading_warmup = warmup
                        model = Qwen3_5ForConditionalGeneration.from_pretrained(
                            None, state_dict=state, **kwargs
                        ).eval()
                else:
                    model = Qwen3_5ForConditionalGeneration.from_pretrained(self.model_id, **kwargs).eval()
                if len(tokenizer) > model.get_input_embeddings().num_embeddings:
                    raise RuntimeError("Tokenizer vocabulary exceeds pinned model embeddings")
                self.tokenizer, self.model = tokenizer, model
                if self.quantization == "nf4":
                    # Drop unoccupied loading reservations before first generation.
                    torch.cuda.empty_cache()
            except Exception as error:
                # A retained exception (e.g. an MCP error/log record) otherwise
                # keeps the partial model alive through dependency frame locals.
                _clear_error_frames(error)
                model, tokenizer = None, None
                self.unload()
                raise

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

    @_inference_operation
    def score(self, candidates, left="", right="", intent=""):
        import torch

        scores = []
        with self.lock, torch.inference_mode():
            self.load()
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

    @_inference_operation
    def generate(self, left, right, intent, target, count=8):
        if not 1 <= count <= 12:
            raise ValueError("count must be between 1 and 12")
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
        with self.lock, torch.inference_mode():
            self.load()
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
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

    def _next_token_options(self, prefix_ids, paths, top_k, microbatch_size):
        """Original full-vocabulary probabilities; filtering does not renormalize top-k."""
        import torch

        result = []
        eos = self.model.generation_config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        excluded = set(self.tokenizer.all_special_ids) - eos_ids
        pad = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        for start in range(0, len(paths), microbatch_size):
            sequences = [list(prefix_ids) + list(path) for path in paths[start:start + microbatch_size]]
            width = max(map(len, sequences))
            ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=self.model.device)
            mask = torch.zeros_like(ids)
            for i, sequence in enumerate(sequences):
                ids[i, -len(sequence):] = torch.tensor(sequence, device=self.model.device)
                mask[i, -len(sequence):] = 1
            output = self.model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
            logprobs = output.logits[:, -1].float().log_softmax(-1)
            for row in logprobs:
                # Probabilities were normalized BEFORE removing nontext controls.
                if excluded:
                    row[list(excluded)] = -float("inf")
                values, tokens = row.topk(min(top_k, row.numel() - len(excluded)))
                result.append(list(zip(tokens.tolist(), values.tolist())))
            del output, logprobs, ids, mask
        return result

    @_inference_operation
    def search(self, *, left, intent, rhyme_weight, beam_width=20, branch_top_k=20,
               max_new_tokens=12, min_new_tokens=2, strength=2.0, length_normalization=0.0,
               limit=10, microbatch_size=1):
        from .search import beam_search, validate_search

        validate_search(beam_width, branch_top_k, max_new_tokens, min_new_tokens,
                        strength, length_normalization, limit, microbatch_size)
        self.load()
        messages = [
            {"role": "system", "content": "You are an English songwriter. Complete the lyric naturally. "
             "Write only original English lyric text, with no explanations or quotation marks. "
             "Continue directly after the supplied lyric prefix; do not repeat it."},
            {"role": "user", "content": "Continue the lyric." + (" Meaning and mood: " + intent if intent else "")},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        ) + left
        prefix = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(prefix) + max_new_tokens > MAX_INPUT_TOKENS:
            raise ValueError(f"Search prompt and token budget exceed {MAX_INPUT_TOKENS} tokens")
        eos = self.model.generation_config.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
        result = beam_search(
            lambda paths: self._next_token_options(prefix, paths, branch_top_k, microbatch_size),
            lambda ids: self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
            rhyme_weight, eos_ids=eos_ids, beam_width=beam_width, branch_top_k=branch_top_k,
            max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens, strength=strength,
            length_normalization=length_normalization, limit=limit,
        )
        result["search"].update({"microbatch_size": microbatch_size, "prefix_token_count": len(prefix),
                                "conditioning_token_ids": prefix,
                                "nontext_special_tokens_excluded": True,
                                "cache_strategy": "Full-prefix forwards; no persistent beam KV copies"})
        return result


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
