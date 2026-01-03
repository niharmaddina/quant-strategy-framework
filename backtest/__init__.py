"""Backtesting engine and visualization."""

from .engine import BacktestEngine

# Lazy import visualizers to avoid matplotlib dependency for tests
def __getattr__(name):
    if name == 'BacktestVisualizer':
        try:
            from .visualizer import BacktestVisualizer
            return BacktestVisualizer
        except ImportError:
            raise ImportError(
                "BacktestVisualizer requires matplotlib. "
                "Install with: pip install matplotlib"
            )
    elif name == 'ChartEngine':
        try:
            from .chart_engine import ChartEngine
            return ChartEngine
        except ImportError:
            raise ImportError(
                "ChartEngine requires matplotlib. "
                "Install with: pip install matplotlib"
            )
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['BacktestEngine', 'BacktestVisualizer', 'ChartEngine']

