import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Final

import torch
from transformers import LogitsProcessor

logger = logging.getLogger("OmniCognitiveUltraV4")


@dataclass(slots=True)
class UltraTelemetry:
    """Enterprise-grade telemetry layout optimized for torch.compile tracking."""

    entropies: List[float] = field(default_factory=list)
    varentropies: List[float] = field(default_factory=list)
    effective_temps: List[float] = field(default_factory=list)
    surviving_tokens: List[int] = field(default_factory=list)
    chaos_streaks: List[int] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        if not self.entropies:
            return {}

        n: Final[int] = len(self.entropies)
        sorted_tokens = sorted(self.surviving_tokens)

        p50 = sorted_tokens[int(n * 0.50)]
        p95 = sorted_tokens[int(n * 0.95)] if n > 19 else sorted_tokens[-1]
        p99 = sorted_tokens[int(n * 0.99)] if n > 99 else sorted_tokens[-1]

        mean_h = sum(self.entropies) / n
        mean_v = sum(self.varentropies) / n

        quadrant = "Stable-Convergent"

        if mean_h > 2.2 and mean_v > 1.2:
            quadrant = "High-Chaos-Exploratory"
        elif mean_h > 2.2:
            quadrant = "High-Entropy-Uniform"
        elif mean_v > 1.2:
            quadrant = "High-Variance-Bimodal"

        return {
            "total_steps": n,
            "mean_entropy_nats": round(mean_h, 4),
            "mean_varentropy": round(mean_v, 4),
            "mean_effective_temp": round(
                sum(self.effective_temps) / n, 4
            ),
            "max_chaos_streak": (
                max(self.chaos_streaks)
                if self.chaos_streaks
                else 0
            ),
            "structural_quadrant": quadrant,
            "tokens_p50": p50,
            "tokens_p95": p95,
            "tokens_p99": p99,
        }


class OmniCognitiveLogitsProcessor(LogitsProcessor):
    """V4 Ultra unified compiler-safe GPU sampling engine."""

    def __init__(
        self,
        prompt_len: int,
        base_threshold: float = 1.2,
        alpha: float = 0.08,
        varentropy_weight: float = 0.15,
        k_min: int = 2,
        k_max: int = 32,
        min_p: float = 0.08,
        temp_min: float = 0.35,
        cooling_rate: float = 0.30,
        decay_lambda: float = 0.45,
        repetition_penalty: float = 1.15,
        frequency_penalty: float = 0.15,
        presence_penalty: float = 0.20,
        dry_n: int = 4,
        chaos_streak_limit: int = 3,
        record_telemetry: bool = True,
    ) -> None:

        self.prompt_len = prompt_len
        self.base_threshold: Final[float] = base_threshold
        self.alpha: Final[float] = alpha
        self.gamma: Final[float] = varentropy_weight

        self.k_min: Final[int] = k_min
        self.k_max: Final[int] = k_max

        self.min_p: Final[float] = min_p
        self.temp_min: Final[float] = temp_min
        self.beta: Final[float] = cooling_rate
        self.decay_lambda: Final[float] = decay_lambda

        self.rep_penalty: Final[float] = repetition_penalty
        self.freq_penalty: Final[float] = frequency_penalty
        self.pres_penalty: Final[float] = presence_penalty

        self.dry_n: Final[int] = dry_n
        self.chaos_limit: Final[int] = chaos_streak_limit

        self.record_telemetry: Final[bool] = record_telemetry
        self.telemetry = UltraTelemetry()

        self.current_chaos_streaks: torch.Tensor = torch.empty(
            0, dtype=torch.long
        )

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:

        device: Final[torch.device] = scores.device
        original_dtype: Final[torch.dtype] = scores.dtype

        batch_size, vocab_size = scores.shape

        if (
            self.current_chaos_streaks.device != device
            or self.current_chaos_streaks.shape[0] != batch_size
        ):
            self.current_chaos_streaks = torch.zeros(
                batch_size,
                dtype=torch.long,
                device=device,
            )

        scores_fp32 = scores.float()

        # 1. Vectorized 2D Matrix Penalty Engine

        history = input_ids[:, self.prompt_len:]

        if history.shape[1] > 0:
            frequencies = torch.zeros(
                (batch_size, vocab_size),
                dtype=torch.float32,
                device=device,
            )

            ones = torch.ones_like(
                history,
                dtype=torch.float32,
                device=device,
            )

            frequencies.scatter_add_(1, history, ones)

            has_appeared = frequencies > 0.0

            # Repetition penalty

            scores_fp32 = torch.where(
                has_appeared,
                torch.where(
                    scores_fp32 > 0.0,
                    scores_fp32 / self.rep_penalty,
                    scores_fp32 * self.rep_penalty,
                ),
                scores_fp32,
            )

            # Frequency and presence penalties

            scores_fp32 -= (
                frequencies * self.freq_penalty
                + has_appeared.float() * self.pres_penalty
            )

        # 2. Entropy and Varentropy

        log_probs = torch.log_softmax(scores_fp32, dim=-1)
        probs = torch.exp(log_probs)

        valid_mask = probs > 0.0

        surprisal = torch.where(
            valid_mask,
            -log_probs,
            torch.zeros_like(log_probs),
        )

        entropy = torch.sum(
            probs * surprisal,
            dim=-1,
        )

        centered = torch.where(
            valid_mask,
            surprisal - entropy.unsqueeze(-1),
            torch.zeros_like(surprisal),
        )

        varentropy = torch.sum(
            probs * centered.pow(2),
            dim=-1,
        )

        # 3. Dynamic Instability Calculation

        gen_steps = max(
            1,
            input_ids.shape[-1] - self.prompt_len,
        )

        dynamic_threshold = (
            self.base_threshold
            + self.alpha * math.log1p(gen_steps)
        )

        excess_entropy = torch.clamp(
            entropy - dynamic_threshold,
            min=0.0,
        )

        instability_index = (
            excess_entropy
            * (1.0 + self.gamma * varentropy)
        )

        # 4. Global Chaos Tracking

        chaos_detected = excess_entropy > 0.5

        self.current_chaos_streaks = torch.where(
            chaos_detected,
            self.current_chaos_streaks + 1,
            torch.zeros_like(self.current_chaos_streaks),
        )

        chaos_modifier = torch.where(
            self.current_chaos_streaks >= self.chaos_limit,
            torch.full_like(
                instability_index,
                float(self.dry_n),
            ),
            torch.ones_like(instability_index),
        )

        # 5. Continuous Temperature Modulation

        effective_temp = torch.clamp(
            1.0 / (
                1.0
                + self.beta
                * instability_index
                * chaos_modifier
            ),
            min=self.temp_min,
            max=1.0,
        )

        scaled_scores = (
            scores_fp32 / effective_temp.unsqueeze(-1)
        )

        # 6. Adaptive Top-K and Min-P Filtering

        max_probs, _ = probs.max(
            dim=-1,
            keepdim=True,
        )

        min_p_mask = probs >= (
            max_probs * self.min_p
        )

        uncertain = excess_entropy > 0.0

        k_adaptive = torch.clamp(
            torch.round(
                self.k_min
                + (self.k_max - self.k_min)
                * torch.exp(
                    -self.decay_lambda * instability_index
                )
            ).long(),
            min=self.k_min,
            max=min(self.k_max, vocab_size),
        )

        row_k = torch.where(
            uncertain,
            k_adaptive,
            torch.full_like(k_adaptive, vocab_size),
        )

        max_k_needed = (
            int(row_k.max().item())
            if row_k.numel() > 0
            else vocab_size
        )

        _, top_idx = torch.topk(
            scaled_scores,
            k=max_k_needed,
            dim=-1,
        )

        ranks = torch.arange(
            max_k_needed,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        topk_mask = torch.zeros_like(
            scaled_scores,
            dtype=torch.bool,
            device=device,
        ).scatter_(
            -1,
            top_idx,
            ranks < row_k.unsqueeze(1),
        )

        final_keep_mask = torch.where(
            uncertain.unsqueeze(-1),
            topk_mask & min_p_mask,
            min_p_mask,
        )

        # Preserve the highest-scoring token

        final_keep_mask.scatter_(
            -1,
            torch.argmax(
                scaled_scores,
                dim=-1,
                keepdim=True,
            ),
            True,
        )

        output_scores = scaled_scores.masked_fill(
            ~final_keep_mask,
            float("-inf"),
        )

        # 7. Optional Telemetry

        if self.record_telemetry:
            self.telemetry.entropies.append(
                float(entropy[0].item())
            )

            self.telemetry.varentropies.append(
                float(varentropy[0].item())
            )

            self.telemetry.effective_temps.append(
                float(effective_temp[0].item())
            )

            self.telemetry.surviving_tokens.append(
                int(final_keep_mask[0].sum().item())
            )

            self.telemetry.chaos_streaks.append(
                int(self.current_chaos_streaks[0].item())
            )

        return output_scores.to(original_dtype)


def create_ultimate_processor(
    **kwargs,
) -> OmniCognitiveLogitsProcessor:
    return OmniCognitiveLogitsProcessor(**kwargs)
