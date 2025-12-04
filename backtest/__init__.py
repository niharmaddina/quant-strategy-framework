"""Backtesting engine and visualization."""

from .engine import BacktestEngine

# Lazy import visualizer to avoid matplotlib dependency for tests
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
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['BacktestEngine', 'BacktestVisualizer']

