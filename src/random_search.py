"""
Random Search module for adversarial suffix optimization.

Adapted from llm-adaptive-attacks (Andriushchenko et al., ICLR 2025)
for the DTA framework.  Supports both local-model (white-box) and
API-model (black-box) modes.

White-box mode:
    Uses a HuggingFace model's ``generate(output_scores=True)`` to
    extract logprobs — no gradient computation needed.

API mode (Strategy 5 — transfer attack):
    Uses an OpenAI-compatible client that returns ``top_logprobs``.

Usage:
    from random_search import RandomSearchModule

    rs = RandomSearchModule(model=llm, tokenizer=tok, device="cuda:0")
    suffix_str, suffix_ids, logprob = rs.random_search(
        base_prompt="...",
        target_token="Sure",
    )
"""

import random
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class RandomSearchModule:
    """Random Search for adversarial suffix optimization."""

    def __init__(
        self,
        model=None,
        tokenizer=None,
        device: str = "cuda:0",
        target_token: str = "Sure",
        n_tokens_adv: int = 25,
        n_tokens_change_max: int = 4,
        n_iterations: int = 500,
        use_schedule: bool = True,
        early_stop_prob: float = 0.1,
        # API mode parameters (Strategy 5)
        api_client=None,
        api_model_name: Optional[str] = None,
    ):
        """
        Args:
            model: A HuggingFace model (or PEFT-wrapped model) for white-box
                   logprob extraction.  Set to ``None`` for API-only mode.
            tokenizer: The tokenizer matching ``model``.
            device: Device string, e.g. ``"cuda:0"``.
            target_token: Default target token to maximise logprob of.
            n_tokens_adv: Number of tokens in the adversarial suffix.
            n_tokens_change_max: Max tokens to mutate per iteration.
            n_iterations: Maximum RS iterations.
            use_schedule: Whether to use adaptive mutation scheduling.
            early_stop_prob: Stop when target-token probability exceeds this.
            api_client: An ``openai.OpenAI`` compatible client for API mode.
            api_model_name: Model name for API calls (e.g. ``"gpt-3.5-turbo"``).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.target_token = target_token
        self.n_tokens_adv = n_tokens_adv
        self.n_tokens_change_max = n_tokens_change_max
        self.n_iterations = n_iterations
        self.use_schedule = use_schedule
        self.early_stop_prob = early_stop_prob

        # API mode
        self.api_client = api_client
        self.api_model_name = api_model_name
        self.api_supports_logprobs = True
        self._warned_logprobs_fallback = False

        # Derive vocab size
        if tokenizer is not None:
            self.vocab_size = len(tokenizer.get_vocab())
        else:
            self.vocab_size = 100256  # GPT-4 default

        # Determine mode
        self.is_api_mode = model is None and api_client is not None

    # ------------------------------------------------------------------
    # Logprob extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_logprob(
        self,
        full_prompt: str,
        target_token: Optional[str] = None,
    ) -> float:
        """
        Extract the log-probability of ``target_token`` being the first
        generated token.

        Returns ``-inf`` if the token is not in the vocabulary.
        """
        if target_token is None:
            target_token = self.target_token

        if self.is_api_mode:
            return self._extract_logprob_api(full_prompt, target_token)

        return self._extract_logprob_local(full_prompt, target_token)

    def _extract_logprob_local(self, full_prompt: str, target_token: str) -> float:
        was_training = self.model.training
        self.model.eval()

        input_ids = self.tokenizer(
            full_prompt, return_tensors="pt"
        ).input_ids.to(self.device)

        output = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=1,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
        log_probs = F.log_softmax(output.scores[0], dim=-1)  # (1, V)

        # Check both "Sure" and " Sure" variants
        candidates = [target_token, " " + target_token]
        best = -np.inf
        for tok in candidates:
            ids = self.tokenizer.encode(tok, add_special_tokens=False)
            for tid in ids:
                val = log_probs[0, tid].item()
                if val > best:
                    best = val

        if was_training:
            self.model.train()
        return best

    def _extract_logprob_api(self, full_prompt: str, target_token: str) -> float:
        """Extract logprob via an OpenAI-compatible API (top_logprobs=20)."""
        for attempt in range(5):
            try:
                if self.api_supports_logprobs:
                    response = self.api_client.chat.completions.create(
                        model=self.api_model_name,
                        messages=[{"role": "user", "content": full_prompt}],
                        max_tokens=1,
                        temperature=0,
                        logprobs=True,
                        top_logprobs=20,
                        seed=0,
                    )
                    logprob_content = response.choices[0].logprobs.content
                    if not logprob_content:
                        raise ValueError("API returned empty logprobs content.")
                    logprob_dict = {
                        lp.token: lp.logprob
                        for lp in logprob_content[0].top_logprobs
                    }
                    return _extract_logprob_from_dict(logprob_dict, target_token)

                return self._extract_logprob_api_without_logprobs(
                    full_prompt=full_prompt,
                    target_text=target_token,
                    max_tokens=max(
                        4,
                        len(
                            self.tokenizer.encode(
                                " " + target_token,
                                add_special_tokens=False,
                            )
                        )
                        + 2,
                    ),
                )
            except Exception as e:
                if _api_error_indicates_unsupported_logprobs(e):
                    self.api_supports_logprobs = False
                    if not self._warned_logprobs_fallback:
                        print(
                            "[RS] API provider does not support logprobs; "
                            "falling back to text-prefix scoring."
                        )
                        self._warned_logprobs_fallback = True
                    continue
                print(f"API error (attempt {attempt + 1}): {e}")
                time.sleep(10)
        return -np.inf

    @torch.no_grad()
    def extract_logprob_for_sequence(
        self,
        full_prompt: str,
        target_token_ids: List[int],
    ) -> float:
        """
        Compute average log-probability of a multi-token target sequence.
        Used for Strategy 3 (dynamic target RS).
        """
        if self.is_api_mode:
            return self._extract_logprob_for_sequence_api(
                full_prompt, target_token_ids
            )

        was_training = self.model.training
        self.model.eval()

        input_ids = self.tokenizer(
            full_prompt, return_tensors="pt"
        ).input_ids.to(self.device)

        n_target = len(target_token_ids)
        output = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=n_target,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

        total_logprob = 0.0
        n_valid = 0
        for i, tid in enumerate(target_token_ids):
            if i >= len(output.scores):
                break
            lp = F.log_softmax(output.scores[i], dim=-1)
            total_logprob += lp[0, tid].item()
            n_valid += 1

        if was_training:
            self.model.train()
        return total_logprob / max(n_valid, 1)

    def _extract_logprob_for_sequence_api(
        self, full_prompt: str, target_token_ids: List[int]
    ) -> float:
        """Multi-token logprob extraction via API — uses per-token logprobs."""
        n_target = len(target_token_ids)
        for attempt in range(5):
            try:
                if self.api_supports_logprobs:
                    response = self.api_client.chat.completions.create(
                        model=self.api_model_name,
                        messages=[{"role": "user", "content": full_prompt}],
                        max_tokens=n_target,
                        temperature=0,
                        logprobs=True,
                        top_logprobs=20,
                        seed=0,
                    )
                    total = 0.0
                    content_logprobs = response.choices[0].logprobs.content
                    n_valid = min(len(content_logprobs), n_target)
                    for i in range(n_valid):
                        token_lp = content_logprobs[i].logprob
                        total += token_lp
                    return total / max(n_valid, 1)

                return self._extract_logprob_api_without_logprobs(
                    full_prompt=full_prompt,
                    target_text=self.tokenizer.decode(target_token_ids),
                    max_tokens=max(n_target, 8),
                )
            except Exception as e:
                if _api_error_indicates_unsupported_logprobs(e):
                    self.api_supports_logprobs = False
                    if not self._warned_logprobs_fallback:
                        print(
                            "[RS] API provider does not support logprobs; "
                            "falling back to text-prefix scoring."
                        )
                        self._warned_logprobs_fallback = True
                    continue
                print(f"API error (attempt {attempt + 1}): {e}")
                time.sleep(10)
        return -np.inf

    def _extract_logprob_api_without_logprobs(
        self,
        full_prompt: str,
        target_text: str,
        max_tokens: int,
    ) -> float:
        """
        Approximate first-token probability when the provider exposes no logprobs.

        We greedily generate a short continuation and score how much of the
        desired prefix appears at the start of the response. The returned value
        is a pseudo-logprob in ``(-inf, 0]`` so the existing RS schedule and
        early-stop logic can keep working.
        """
        response = self.api_client.chat.completions.create(
            model=self.api_model_name,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=max_tokens,
            temperature=0,
            seed=0,
        )
        generated_text = response.choices[0].message.content or ""
        return _approximate_logprob_from_text_prefix(
            generated_text=generated_text,
            target_text=target_text,
        )

    # ------------------------------------------------------------------
    # Core Random Search
    # ------------------------------------------------------------------

    def random_search(
        self,
        base_prompt: str,
        adv_init: Optional[str] = None,
        target_token: Optional[str] = None,
        dynamic_target: bool = False,
        dynamic_target_fn: Optional[Callable] = None,
        resample_interval: int = 50,
    ) -> Tuple[str, List[int], float]:
        """
        Run random search to find an adversarial suffix.

        Args:
            base_prompt: The (possibly templated) prompt.
            adv_init: Initial adversarial suffix string.  If ``None``, uses
                      ``' !' * n_tokens_adv``.
            target_token: Token whose logprob to maximise.
            dynamic_target: If ``True``, periodically re-sample targets
                            (Strategy 3).
            dynamic_target_fn: ``fn(current_suffix_str) -> (token_ids, score)``
                               called every ``resample_interval`` iterations.
            resample_interval: How often to call ``dynamic_target_fn``.

        Returns:
            ``(best_suffix_str, best_suffix_token_ids, best_logprob)``
        """
        if target_token is None:
            target_token = self.target_token

        # Initialise adversarial suffix
        if adv_init is None:
            adv_init = " !" * self.n_tokens_adv

        adv_tokens = self.tokenizer.encode(adv_init, add_special_tokens=False)
        best_adv_tokens = list(adv_tokens)
        best_adv_str = self.tokenizer.decode(best_adv_tokens)
        best_logprob = -np.inf

        # Dynamic target state
        current_target_token_ids = None  # multi-token target for Strategy 3
        use_sequence_logprob = False

        n_tokens_change = self.n_tokens_change_max

        for it in range(1, self.n_iterations + 1):
            # --- Strategy 3: periodic re-sampling ---
            if (
                dynamic_target
                and dynamic_target_fn is not None
                and it % resample_interval == 0
            ):
                new_target_ids, new_score = dynamic_target_fn(best_adv_str)
                if new_target_ids is not None and len(new_target_ids) > 0:
                    current_target_token_ids = new_target_ids
                    use_sequence_logprob = True
                    print(
                        f"[RS iter {it}] Dynamic target updated "
                        f"(score={new_score:.3f}, "
                        f"n_tokens={len(current_target_token_ids)})"
                    )

            # Construct message
            adv_str = self.tokenizer.decode(adv_tokens)
            msg = base_prompt + " " + adv_str

            # Evaluate
            if use_sequence_logprob and current_target_token_ids is not None:
                logprob = self.extract_logprob_for_sequence(
                    msg, current_target_token_ids
                )
            else:
                logprob = self.extract_logprob(msg, target_token)

            # Update best
            if logprob > best_logprob:
                best_logprob = logprob
                best_adv_tokens = list(adv_tokens)
                best_adv_str = adv_str
            else:
                # Revert
                adv_tokens = list(best_adv_tokens)

            # Logging
            if it % 50 == 0 or it == 1:
                prob = np.exp(best_logprob) if best_logprob > -50 else 0.0
                print(
                    f"[RS iter {it}/{self.n_iterations}] "
                    f"best_logprob={best_logprob:.3f} "
                    f"prob={prob:.5f} "
                    f"n_change={n_tokens_change}"
                )

            # Early stopping
            if best_logprob > -50 and np.exp(best_logprob) > self.early_stop_prob:
                print(
                    f"[RS] Early stopping at iter {it}: "
                    f"prob={np.exp(best_logprob):.4f} > {self.early_stop_prob}"
                )
                break

            # --- Mutation ---
            if self.use_schedule:
                prob = np.exp(best_logprob) if best_logprob > -50 else 0.0
                n_tokens_change = self.schedule_n_to_change_prob(
                    self.n_tokens_change_max, prob
                )

            # Randomly substitute tokens
            if len(adv_tokens) == 0:
                break
            start_pos = random.randint(0, max(0, len(adv_tokens) - 1))
            new_tokens = [
                random.randint(0, self.vocab_size - 1)
                for _ in range(n_tokens_change)
            ]
            adv_tokens = (
                adv_tokens[:start_pos]
                + new_tokens
                + adv_tokens[start_pos + n_tokens_change :]
            )

        best_adv_str = self.tokenizer.decode(best_adv_tokens)
        print(
            f"[RS] Finished: best_logprob={best_logprob:.3f} "
            f"prob={np.exp(best_logprob) if best_logprob > -50 else 0.0:.5f} "
            f"suffix='{best_adv_str[:80]}...'"
        )
        return best_adv_str, best_adv_tokens, best_logprob

    # ------------------------------------------------------------------
    # Conversion utilities
    # ------------------------------------------------------------------

    def suffix_tokens_to_logits(
        self,
        token_ids: List[int],
        vocab_size: int,
        suffix_max_length: int,
        peak_value: float = 5.0,
    ) -> torch.Tensor:
        """
        Convert a token ID sequence (from RS) into **smooth** logits
        suitable for DTA suffix initialisation.

        Instead of hard one-hot (which causes gradient saturation in DTA),
        this produces a soft distribution centred on the RS-found tokens
        with a controllable ``peak_value``.  The background is small
        Gaussian noise so that the DTA gradient optimiser can explore
        neighbouring tokens.

        Args:
            token_ids: Token IDs from the RS phase.
            vocab_size: Vocabulary size of the model.
            suffix_max_length: Target suffix length for DTA.
            peak_value: Logit value at the RS-found token.  Background
                noise has std ≈ 1.0.  A ``peak_value`` of 5.0 gives the
                RS token ~99 % of the softmax mass while keeping gradients
                alive for nearby tokens.

        Returns:
            Tensor of shape ``(1, suffix_max_length, vocab_size)``.
        """
        ids = list(token_ids[:suffix_max_length])

        # Start from small Gaussian noise (like model output logits)
        logits = torch.randn(1, suffix_max_length, vocab_size) * 1.0

        # Boost the RS-found tokens
        for i, tid in enumerate(ids):
            if 0 <= tid < vocab_size:
                logits[0, i, tid] = peak_value

        return logits

    # ------------------------------------------------------------------
    # Adaptive scheduling (ported from llm-adaptive-attacks/utils.py)
    # ------------------------------------------------------------------

    @staticmethod
    def schedule_n_to_change_prob(max_n_to_change: int, prob: float) -> int:
        """
        Piece-wise constant schedule for mutation magnitude based on the
        current probability of the target token.

        Ported from ``llm-adaptive-attacks/utils.py:schedule_n_to_change_prob``.
        """
        if 0 <= prob <= 0.01:
            n = max_n_to_change
        elif prob <= 0.1:
            n = max_n_to_change // 2
        elif prob <= 1.0:
            n = max_n_to_change // 4
        else:
            n = max_n_to_change // 4
        return max(n, 1)

    @staticmethod
    def schedule_n_to_change_fixed(max_n_to_change: int, it: int) -> int:
        """
        Piece-wise constant schedule based on iteration count.

        Ported from ``llm-adaptive-attacks/utils.py:schedule_n_to_change_fixed``.
        """
        if it <= 10:
            n = max_n_to_change
        elif it <= 25:
            n = max_n_to_change // 2
        elif it <= 50:
            n = max_n_to_change // 4
        elif it <= 100:
            n = max_n_to_change // 8
        elif it <= 500:
            n = max_n_to_change // 16
        else:
            n = max_n_to_change // 32
        return max(n, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_logprob_from_dict(
    logprob_dict: Dict[str, float],
    target_token: str,
) -> float:
    """
    Extract the logprob of ``target_token`` from a logprob dictionary.

    Checks both ``target_token`` and ``" " + target_token`` to handle
    tokenisation variants.

    Ported from ``llm-adaptive-attacks/utils.py:extract_logprob``.
    """
    candidates = []
    if " " + target_token in logprob_dict:
        candidates.append(logprob_dict[" " + target_token])
    if target_token in logprob_dict:
        candidates.append(logprob_dict[target_token])
    return max(candidates) if candidates else -np.inf


def _api_error_indicates_unsupported_logprobs(exc: Exception) -> bool:
    message = str(exc).lower()
    if "logprobs" not in message and "top_logprobs" not in message:
        return False
    unsupported_markers = [
        "unexpected keyword argument",
        "unsupported",
        "not support",
        "unknown parameter",
        "extra inputs are not permitted",
        "extra_forbidden",
        "unrecognized",
        "invalid parameter",
    ]
    return any(marker in message for marker in unsupported_markers)


def _approximate_logprob_from_text_prefix(
    generated_text: str,
    target_text: str,
) -> float:
    """
    Convert prefix overlap into a pseudo-logprob.

    Exact prefix match maps to ``0.0`` (probability 1). Partial matches map
    to ``log(prefix_ratio)`` so the existing RS code can still interpret the
    score through ``exp(score)``.
    """
    generated = (generated_text or "").lstrip()
    target = (target_text or "").lstrip()
    if not target:
        return -np.inf

    candidate_targets = [target]
    lowered = target.lower()
    if lowered != target:
        candidate_targets.append(lowered)

    best_ratio = 0.0
    generated_lower = generated.lower()
    for candidate in candidate_targets:
        best_ratio = max(best_ratio, _prefix_match_ratio(generated, candidate))
        if generated_lower != generated:
            best_ratio = max(
                best_ratio,
                _prefix_match_ratio(generated_lower, candidate.lower()),
            )

    if best_ratio <= 0:
        return -np.inf
    return float(np.log(best_ratio))


def _prefix_match_ratio(text: str, prefix: str) -> float:
    if not text or not prefix:
        return 0.0
    if text.startswith(prefix):
        return 1.0

    match_len = 0
    max_len = min(len(text), len(prefix))
    while match_len < max_len and text[match_len] == prefix[match_len]:
        match_len += 1
    return match_len / len(prefix)
