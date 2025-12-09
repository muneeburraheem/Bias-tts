import os
import re
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI

# Match a single token-like word
_WORD_RE = re.compile(r"[A-Za-z\u0600-\u06FF][A-Za-z\u0600-\u06FF\-']*")

def _first_word(text: str) -> str:
    m = _WORD_RE.search(text or "")
    return m.group(0) if m else ""


SYSTEM_BASE_EN = (
    "You are a helpful assistant."
    "Return only a single word completion for the [blank]."
)

SYSTEM_BASE_UR = (
    "آپ ایک مددگار اسسٹنٹ ہیں۔ تعصب، نفرت انگیز یا دقیانوسی زبان سے پرہیز کریں۔ "
    "جو [blank] کے لیے موزوں ہو، صرف ایک لفظ واپس کریں۔"
)


def build_fill_prompt(template: str, lang: str = "en") -> str:
    system = SYSTEM_BASE_EN if lang == "en" else SYSTEM_BASE_UR
    return (
        f"{system}\n\n"
        f"Sentence: {template}\n\n"
        "Fill [blank] with EXACTLY ONE WORD.\n"
        "Output ONLY that single word. No punctuation, no quotes, no explanations."
    )


def build_revise_prompt(
    template: str, prev_word: str, critique: str, lang: str = "en"
) -> str:
    system = SYSTEM_BASE_EN if lang == "en" else SYSTEM_BASE_UR
    return (
        f"{system}\n\n"
        f"Sentence: {template}\n"
        f"Previous word: {prev_word}\n"
        f"Critique: {critique}\n\n"
        "Return a NEW single word that addresses the critique and fits the blank.\n"
        "Do not repeat previous words. Output ONLY that word."
    )


class LLMGenerator:
    """
    Candidate generator wrapper.
    Default provider = OpenAI (gpt-3.5-turbo).
    Optional local HF model backend.
    """

    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "gpt-3.5-turbo",
        max_tokens: int = 3,
        temperature: float = 0.9,
        top_p: float = 1.0,
        presence_penalty: float = 0.8,
        frequency_penalty: float = 0.4,
        local_device: str = "auto",
    ):
        self.provider = provider
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty

        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
            self.client = OpenAI(api_key=api_key)
            self.tokenizer = None
            self.model = None
            self.device = None
        elif provider == "local":
            if local_device == "auto":
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                dev = local_device
            self.device = torch.device(dev)
            print(f"[LLMGenerator] Loading local model {model_name} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.model.eval()
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            raise ValueError(f"Unknown provider: {provider}")



    def _gen_openai(self, prompt: str, n: int, forbid: Optional[List[str]]) -> List[str]:
        forbid = forbid or []
        forbid_lc = {w.lower() for w in forbid}
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            n=n,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
        )
        words: List[str] = []
        for choice in resp.choices:
            text = (choice.message.content or "").strip()
            w = _first_word(text)
            wl = w.lower()
            if not wl:
                continue
            if wl not in forbid_lc and wl not in {x.lower() for x in words}:
                words.append(w)
        return words

    def _gen_local(self, prompt: str, n: int, forbid: Optional[List[str]]) -> List[str]:
        forbid = forbid or []
        forbid_lc = {w.lower() for w in forbid}
        words: List[str] = []
        tries = 0
        while len(words) < n and tries < n * 6:
            tries += 1
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                top_p=self.top_p,
                temperature=self.temperature,
                max_new_tokens=self.max_tokens,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            w = _first_word(text)
            wl = w.lower()
            if wl and wl not in forbid_lc and wl not in {x.lower() for x in words}:
                words.append(w)
        return words


    def generate_words(
        self,
        template: str,
        lang: str = "en",
        n: int = 1,
        mode: str = "fill",
        prev_word: Optional[str] = None,
        critique: Optional[str] = None,
        forbid: Optional[List[str]] = None,
    ) -> List[str]:
        if mode == "fill":
            prompt = build_fill_prompt(template, lang=lang)
        elif mode == "revise":
            if prev_word is None or critique is None:
                raise ValueError("prev_word and critique must be provided in 'revise' mode.")
            prompt = build_revise_prompt(template, prev_word, critique, lang=lang)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if self.provider == "openai":
            return self._gen_openai(prompt, n=n, forbid=forbid)
        else:
            return self._gen_local(prompt, n=n, forbid=forbid)