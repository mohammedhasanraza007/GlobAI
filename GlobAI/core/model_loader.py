"""
core/model_loader.py
--------------------
Bounded local LLM loader for transformers-compatible instruct models.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import threading
import types
from pathlib import Path
from typing import Any, Tuple

import torch

from core.device_resolver import resolve_device
from bootstrap.model_cache import candidate_model_roots
from core.memory_manager import (
    DuplicateModelLoadError,
    MemoryManager,
    ModelCleanupError,
    TokenLimitError,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class ModelNotLoadedError(RuntimeError):
    pass


class ModelValidationError(RuntimeError):
    pass


class ModelLoader:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device_preference: str = "cpu",
        cache_dir: str = "model_checkpoints",
        keep_loaded: bool = True,
        max_input_tokens: int = 1536,
        local_files_only: bool = True,
        model_role: str = "rag",
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.keep_loaded = keep_loaded
        self.max_input_tokens = max_input_tokens
        self.local_files_only = local_files_only
        self.model_role = model_role
        self.device, self.device_kind = resolve_device(device_preference)
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded_model_id: str | None = None
        self._model_source: str | None = None
        self._lock = threading.RLock()

    def _torch_dtype(self) -> torch.dtype:
        if self.device_kind == "cuda":
            return torch.float16
        return torch.float32

    def _patch_directml_llama_mask(self, model: Any) -> None:
        if self.device_kind != "directml":
            return
        llama_model = getattr(model, "model", None)
        if llama_model is None or llama_model.__class__.__name__ != "LlamaModel":
            return

        try:
            from transformers.models.llama.modeling_llama import (
                AttentionMaskConverter,
                LlamaRotaryEmbedding,
            )
        except Exception:
            return

        if not getattr(LlamaRotaryEmbedding, "_nexarag_directml_patch", False):
            original_forward = LlamaRotaryEmbedding.forward

            @torch.no_grad()
            def _directml_rotary_forward(rotary_self: Any, x: torch.Tensor, position_ids: torch.Tensor):
                if x.device.type in {"cpu", "cuda"}:
                    return original_forward(rotary_self, x, position_ids)

                inv_freq_expanded = rotary_self.inv_freq[None, :, None].float().expand(
                    position_ids.shape[0],
                    -1,
                    1,
                )
                position_ids_expanded = position_ids[:, None, :].float()
                with torch.autocast(device_type="cpu", enabled=False):
                    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
                    emb = torch.cat((freqs, freqs), dim=-1)
                    cos = emb.cos()
                    sin = emb.sin()
                return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

            LlamaRotaryEmbedding.forward = _directml_rotary_forward
            LlamaRotaryEmbedding._nexarag_directml_patch = True

        def _directml_update_causal_mask(
            patched_self: Any,
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_seen_tokens: int,
        ) -> torch.Tensor | None:
            if patched_self.config._attn_implementation == "flash_attention_2":
                if attention_mask is not None and 0.0 in attention_mask:
                    return attention_mask
                return None

            if patched_self.config._attn_implementation == "sdpa":
                if AttentionMaskConverter._ignore_causal_mask_sdpa(
                    attention_mask,
                    inputs_embeds=input_tensor,
                    past_key_values_length=past_seen_tokens,
                ):
                    return None

            dtype, device = input_tensor.dtype, input_tensor.device
            min_dtype = torch.finfo(dtype).min
            sequence_length = input_tensor.shape[1]
            if hasattr(getattr(patched_self.layers[0], "self_attn", {}), "past_key_value"):
                target_length = patched_self.config.max_position_embeddings
            else:
                target_length = (
                    attention_mask.shape[-1]
                    if isinstance(attention_mask, torch.Tensor)
                    else past_seen_tokens + sequence_length + 1
                )

            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            visible = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask * visible.to(dtype=dtype)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)

            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                if attention_mask.dim() == 2:
                    mask_length = attention_mask.shape[-1]
                    padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                    padding_mask = padding_mask == 0
                    causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                        padding_mask,
                        min_dtype,
                    )
                elif attention_mask.dim() == 4:
                    if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                        offset = cache_position[0]
                    else:
                        offset = 0
                    mask_shape = attention_mask.shape
                    mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                    causal_mask[
                        : mask_shape[0],
                        : mask_shape[1],
                        offset : mask_shape[2] + offset,
                        : mask_shape[3],
                    ] = mask_slice

            return causal_mask

        llama_model._update_causal_mask = types.MethodType(_directml_update_causal_mask, llama_model)
        logger.info("[LOADER] Applied DirectML Llama causal-mask compatibility patch.")

    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def _candidate_model_roots(self) -> list[Path]:
        return candidate_model_roots(
            Path(self.cache_dir),
            self.model_id,
            preferred_role=self.model_role,
        )

    @staticmethod
    def _hash_file(path: Path, algorithm: str = "sha256") -> str:
        digest = hashlib.new(algorithm)
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _validate_checksums(self, root: Path) -> None:
        checksums = root / "checksums.json"
        if not checksums.exists():
            return
        data = json.loads(checksums.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ModelValidationError("checksums.json must contain an object.")
        for relative, expected in data.items():
            target = root / str(relative)
            if not target.exists() or not target.is_file():
                raise ModelValidationError(f"Checksum target missing: {relative}")
            algorithm = "sha256"
            expected_hash = str(expected)
            if isinstance(expected, dict):
                algorithm = str(expected.get("algorithm", "sha256"))
                expected_hash = str(expected.get("hash", ""))
            actual = self._hash_file(target, algorithm=algorithm)
            if actual.lower() != expected_hash.lower():
                raise ModelValidationError(f"Checksum mismatch: {relative}")

    def _validate_model_source(self) -> str:
        for root in self._candidate_model_roots():
            if not root.is_dir():
                continue
            config_path = root / "config.json"
            if not config_path.exists():
                logger.warning("[LOADER] Candidate missing config.json: %s", root)
                continue
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ModelValidationError(f"Invalid config.json in {root}: {exc}") from exc
            if not isinstance(config, dict) or not config:
                raise ModelValidationError(f"config.json is empty or invalid in {root}")

            tokenizer_files = (
                "tokenizer.json",
                "tokenizer.model",
                "vocab.json",
                "spiece.model",
            )
            if not any((root / name).exists() for name in tokenizer_files):
                logger.warning("[LOADER] Candidate missing tokenizer files: %s", root)
                continue

            self._validate_checksums(root)
            logger.info("[LOADER] Validated local model folder: %s", root)
            return str(root)

        raise ModelValidationError(
            f"No valid local model folder found for '{self.model_id}' in cache_dir='{self.cache_dir}'."
        )

    def load(self) -> Tuple[Any, Any]:
        with self._lock:
            if self.is_loaded() and self._loaded_model_id == self.model_id:
                raise DuplicateModelLoadError(f"Model already loaded: {self.model_id}")
            if self.is_loaded():
                self.unload()

            model_source = self._validate_model_source()
            before_guard = MemoryManager.snapshot()
            logger.info("[LOADER] Loading model %s from %s on %s", self.model_id, model_source, self.device_kind)
            logger.info("[LOADER] Memory before load: %s", before_guard)

            model = None
            tokenizer = None
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    model_source,
                    cache_dir=self.cache_dir,
                    local_files_only=True,
                    trust_remote_code=False,
                )
                tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

                model_kwargs: dict[str, Any] = {
                    "cache_dir": self.cache_dir,
                    "local_files_only": True,
                    "trust_remote_code": False,
                    "low_cpu_mem_usage": True,
                    "torch_dtype": self._torch_dtype(),
                }

                model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
                model = model.to(self.device)
                self._patch_directml_llama_mask(model)
                model.eval()

                self._model = model
                self._tokenizer = tokenizer
                self._loaded_model_id = self.model_id
                self._model_source = model_source
                logger.info("[LOADER] Model loaded. Memory after load: %s", MemoryManager.snapshot())
                return self._model, self._tokenizer
            except Exception:
                logger.exception("[LOADER] Model load failed.")
                if model is not None:
                    del model
                if tokenizer is not None:
                    del tokenizer
                self._model = None
                self._tokenizer = None
                self._loaded_model_id = None
                self._model_source = None
                MemoryManager.hard_cleanup("failed text model load")
                raise

    def unload(self) -> None:
        with self._lock:
            before = MemoryManager.snapshot()
            logger.info("[LOADER] Unload requested for %s. Before: %s", self._loaded_model_id, before)
            had_model = self._model is not None or self._tokenizer is not None
            model = self._model
            tokenizer = self._tokenizer
            self._model = None
            self._tokenizer = None
            self._loaded_model_id = None
            self._model_source = None
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            after = MemoryManager.hard_cleanup("text model unload")
            if had_model and MemoryManager.memory_increased(before, after):
                logger.warning("[LOADER] Memory increased after unload; running emergency cleanup.")
                after = MemoryManager.hard_cleanup("text model emergency cleanup")
                if MemoryManager.memory_increased(before, after):
                    raise ModelCleanupError("Memory did not return to a safe level after text model unload.")
            logger.info("[LOADER] Model unloaded. After: %s", after)

    def cleanup_after_generate(self) -> None:
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def switch_model(self, model_id: str, model_role: str | None = None) -> None:
        model_id = str(model_id or "").strip()
        if not model_id:
            raise ValueError("model_id is required.")
        with self._lock:
            if self.is_loaded():
                self.unload()
            self.model_id = model_id
            if model_role:
                self.model_role = model_role
            logger.info("[LOADER] Model target set to %s. Explicit load required.", model_id)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 384,
        temperature: float = 0.2,
        max_time: float | None = None,
    ) -> str:
        with self._lock:
            if not self.is_loaded():
                raise ModelNotLoadedError("Text model not loaded. Load it first.")
            model, tokenizer = self._model, self._tokenizer
            inputs: dict[str, Any] = {}
            tokenized = None
            output_ids = None
            new_tokens = None
            try:
                tokenized = tokenizer(prompt, return_tensors="pt", truncation=False)
                prompt_tokens = int(tokenized["input_ids"].shape[1])
                if prompt_tokens > self.max_input_tokens:
                    raise TokenLimitError(
                        f"Input has {prompt_tokens} tokens; max_input_tokens is {self.max_input_tokens}."
                    )
                inputs = {name: tensor.to(self.device) for name, tensor in tokenized.items()}

                with torch.no_grad():
                    generation_kwargs: dict[str, Any] = {
                        **inputs,
                        "max_new_tokens": max_tokens,
                        "do_sample": temperature > 0,
                        "pad_token_id": tokenizer.eos_token_id,
                        "repetition_penalty": 1.08,
                    }
                    if temperature > 0:
                        generation_kwargs["temperature"] = temperature
                    else:
                        generation_kwargs["temperature"] = 1.0
                        generation_kwargs["top_p"] = 1.0
                        generation_kwargs["top_k"] = 0
                    if max_time is not None and float(max_time) > 0:
                        generation_kwargs["max_time"] = float(max_time)
                    output_ids = model.generate(**generation_kwargs)

                prompt_len = inputs["input_ids"].shape[1]
                new_tokens = output_ids[0][prompt_len:]
                return tokenizer.decode(new_tokens.detach().cpu(), skip_special_tokens=True).strip()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    logger.error("[LOADER] Out of memory during generation; unloading model.")
                    self.unload()
                raise
            finally:
                for key in list(inputs):
                    del inputs[key]
                if output_ids is not None:
                    del output_ids
                if new_tokens is not None:
                    del new_tokens
                if tokenized is not None:
                    del tokenized
                self.cleanup_after_generate()
                if not self.keep_loaded and self.is_loaded():
                    self.unload()
