# Documentation Index

Keep docs short, current, and aligned with the codebase.

## Core guides

- `docs/MEAN_REVERSION_STRATEGY.md`: Mean reversion indicators + intraday data usage
- `docs/API_BATCHING_GUIDE.md`: Data fetching + batching/rate limit guidance
- `docs/CHART_ENGINE_GUIDE.md`: Chart engine usage
- `universe/UNIVERSE_GUIDE.md`: Universe construction + frozen universe guidance
- `docs/ROADMAP.md`: Short list of remaining improvements (kept current)
- `results/summary.md` and `results/walk_forward.md`: committed outputs of the frozen-universe worked example and its walk-forward evaluation (see README "Results")

## Notes

- For historical backtests, prefer **frozen universe files** (e.g. `universe/universe_mean_reversion_manual_50.txt`) to avoid selection/survivorship-style bias from “today’s” metadata.


