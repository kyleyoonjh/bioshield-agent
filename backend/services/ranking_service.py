"""
Weighted ranking engine.
Reads weights from config/ranking.yaml and computes:

  Final Score = (w_coverage × Coverage) + (w_thermo × Thermo) + (w_ai × AI)

Specificity is a hard filter — any candidate with specificity_valid=False
is excluded before scoring.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "ranking.yaml"
)


class RankingService:
    def __init__(self, config_path: str = _CONFIG_PATH):
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)["ranking"]
            self.w_coverage = cfg["coverage_weight"]
            self.w_thermo   = cfg["thermo_weight"]
            self.w_ai       = cfg["ai_weight"]
        except Exception as exc:
            logger.warning("[ranking] Config load failed (%s) — using defaults", exc)
            self.w_coverage = 0.6
            self.w_thermo   = 0.2
            self.w_ai       = 0.2

    def calculate_final_rankings(
        self,
        candidates: list[dict],
        score_weights: dict[str, float] | None = None,
    ) -> list[dict]:
        """
        1. Drop candidates with specificity_valid=False (hard filter).
        2. Compute Final Score using strategy-specific weights if provided.
        3. Sort descending, assign final_rank starting at 1.
        """
        t0 = time.perf_counter()
        w_coverage = score_weights.get("coverage", self.w_coverage) if score_weights else self.w_coverage
        w_thermo   = score_weights.get("thermo",   self.w_thermo)   if score_weights else self.w_thermo
        w_ai       = score_weights.get("ai",       self.w_ai)       if score_weights else self.w_ai
        logger.info("[PERF][ranking] calculate_final_rankings START | candidates=%d "
                    "weights=(cov=%.2f thermo=%.2f ai=%.2f)%s",
                    len(candidates), w_coverage, w_thermo, w_ai,
                    " [strategy override]" if score_weights else " [yaml default]")

        valid = [c for c in candidates if c.get("specificity_valid", True)]
        filtered = len(candidates) - len(valid)
        if filtered:
            logger.info("[PERF][ranking] specificity_filter | dropped=%d remaining=%d",
                        filtered, len(valid))

        for c in valid:
            c["final_score"] = round(
                w_coverage * c.get("coverage_score", 0.0)
                + w_thermo  * c.get("thermo_score",   0.0)
                + w_ai      * c.get("ai_score",       0.0),
                4,
            )

        valid.sort(key=lambda x: x["final_score"], reverse=True)

        for idx, c in enumerate(valid):
            c["final_rank"] = idx + 1

        top3_scores = [f"{c['final_score']:.2f}" for c in valid[:3]]
        logger.info("[PERF][ranking] DONE | ranked=%d top3_scores=[%s] elapsed=%.3fs",
                    len(valid), ", ".join(top3_scores), time.perf_counter() - t0)
        return valid
