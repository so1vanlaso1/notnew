"""Chat-model backends.

`ChatModel` wraps a HuggingFace causal-LM (or multimodal-it, e.g. Gemma) with:
  * a precision switch — 4bit (NF4) | 8bit (int8) | bf16 (also fp16 / fp32),
  * a robust loader that falls back across model classes / tokenizer vs processor,
  * a chat renderer that tolerates templates with no `system` role (Gemma),
  * `unload()` that actually frees VRAM, so the cascade can keep at most two 4B
    models OR one 8B model resident at any moment.

`StubModel` is a zero-dependency stand-in for wiring tests (no torch needed).
"""

from __future__ import annotations

import gc
import logging
import time

log = logging.getLogger(__name__)

PRECISIONS = ("4bit", "8bit", "bf16", "fp16", "fp32")


class ChatModel:
    def __init__(
        self,
        model_id: str,
        precision: str = "4bit",
        device_map: str = "auto",
        compute_dtype: str = "bfloat16",
        enable_thinking: bool = False,
        label: str = "",
    ):
        import torch  # lazy: keep the package importable without torch
        from transformers import AutoTokenizer

        self._torch = torch
        self.model_id = model_id
        self.label = label or model_id
        self.enable_thinking = enable_thinking

        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")

        has_cuda = torch.cuda.is_available()
        if precision in ("4bit", "8bit") and not has_cuda:
            log.warning("%s: %s needs a CUDA GPU (bitsandbytes); falling back to fp32/CPU.",
                        self.label, precision)
            precision = "fp32"
        self.precision = precision
        compute = getattr(torch, compute_dtype, torch.bfloat16)

        # Tokenizer (fall back to a processor for multimodal repos like Gemma 3n/4).
        self.processor = None
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        except Exception as e:  # noqa: BLE001
            log.warning("%s: AutoTokenizer failed (%s); trying AutoProcessor.", self.label, e)
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token", None):
            try:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception:  # noqa: BLE001
                pass

        quant = None
        load_dtype = compute
        if precision == "4bit":
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute, bnb_4bit_use_double_quant=True,
            )
        elif precision == "8bit":
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(load_in_8bit=True)
        elif precision == "bf16":
            load_dtype = torch.bfloat16
        elif precision == "fp16":
            load_dtype = torch.float16
        elif precision == "fp32":
            load_dtype = torch.float32

        self.model = self._load_model(
            model_id, load_dtype, device_map if has_cuda else "cpu", quant
        )
        self.model.eval()
        log.info("loaded %s (%s, precision=%s)", self.label, model_id, self.precision)

    def _load_model(self, model_id: str, dtype, device_map, quant):
        # `torch_dtype` is ignored by transformers when a quantization_config is
        # present, so passing both is safe.
        kwargs = dict(trust_remote_code=True, torch_dtype=dtype, device_map=device_map)
        if quant is not None:
            kwargs["quantization_config"] = quant
        from transformers import AutoModelForCausalLM
        try:
            return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        except Exception as e:  # noqa: BLE001
            log.warning("%s: AutoModelForCausalLM failed (%s); trying AutoModelForImageTextToText.",
                        self.label, e)
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)

    # ── prompt rendering ──────────────────────────────────────────────────────
    def _apply_template(self, messages: list[dict]) -> str:
        kw = dict(tokenize=False, add_generation_prompt=True)
        try:
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=self.enable_thinking, **kw
            )
        except TypeError:
            # Template doesn't accept `enable_thinking` (most non-Qwen models).
            return self.tokenizer.apply_chat_template(messages, **kw)

    def render(self, system: str, user: str) -> str:
        """Render a system+user chat. If the template rejects a `system` role
        (e.g. Gemma), fold the system text into the user turn."""
        try:
            return self._apply_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}]
            )
        except Exception:  # noqa: BLE001
            return self._apply_template(
                [{"role": "user", "content": f"{system}\n\n{user}"}]
            )

    # ── generation ────────────────────────────────────────────────────────────
    def generate(self, system: str, user: str, max_new_tokens: int = 256,
                 temperature: float = 0.0) -> tuple[str, str, float]:
        """Return (raw_completion, rendered_prompt, elapsed_seconds)."""
        torch = self._torch
        prompt = self.render(system, user)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        do_sample = bool(temperature and temperature > 0.0)
        gen_kwargs = dict(
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=0.9)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        elapsed = time.perf_counter() - t0
        prompt_len = inputs["input_ids"].shape[1]
        text = self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
        return text, prompt, elapsed

    def unload(self) -> None:
        """Drop the model + tokenizer and release VRAM."""
        for attr in ("model", "tokenizer", "processor"):
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except Exception:  # noqa: BLE001
                    pass
        gc.collect()
        try:
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
                self._torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            pass
        log.info("unloaded %s", self.label)


class StubModel:
    """Deterministic, dependency-free model for wiring tests.

    `answer_fn(user_prompt) -> str` decides the raw answer token, letting tests
    drive both the agreement and the disagreement branch without a GPU.
    """

    def __init__(self, label: str, answer_fn):
        self.label = label
        self.model_id = f"stub:{label}"
        self.precision = "stub"
        self._answer_fn = answer_fn

    def generate(self, system: str, user: str, max_new_tokens: int = 256,
                 temperature: float = 0.0) -> tuple[str, str, float]:
        ans = self._answer_fn(user)
        raw = f"ANSWER: {ans}\nWHY: stub explanation from {self.label}."
        prompt = f"[SYSTEM]\n{system}\n\n[USER]\n{user}"
        return raw, prompt, 0.0

    def unload(self) -> None:
        pass
