from src.config import load_config
from src.data_loader import DataLoader, DataLoaderConfig
from src.strategy import SMACross
from src.backtest_runner import BacktestRunner, BacktestConfig
from src.evaluation import summarize_stats
from src.utils import get_logger

logger = get_logger("main")


def main():
    cfg = load_config()

    # --- Data ---
    data_cfg = DataLoaderConfig(
        ticker=cfg["data"]["ticker"],
        start=cfg["data"]["start"],
        end=cfg["data"]["end"],
        timeframe=cfg["data"]["timeframe"],
    )
    loader = DataLoader(data_cfg)
    data = loader.get_price_data()

    # Backtesting.py expects OHLC; ensure columns exist
    data = data[["Open", "High", "Low", "Close", "Volume"]]

    # --- Strategy selection ---
    strategy_name = cfg["strategy"]["name"]
    strategy_params = cfg["strategy"]["params"]

    if strategy_name == "SMACross":
        strategy_cls = SMACross
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # --- Backtest config ---
    bt_cfg = BacktestConfig(
        cash=cfg["backtest"]["cash"],
        commission=cfg["backtest"]["commission"],
        slippage=cfg["backtest"]["slippage"],
    )

    runner = BacktestRunner(
        data=data,
        strategy_cls=strategy_cls,
        bt_config=bt_cfg,
        strategy_params=strategy_params,
    )

    bt, stats = runner.run()

    # --- Evaluation ---
    summarize_stats(stats)

    # Optional: to see the plot interactively
    # bt.plot(open_browser=False)  # or True if you like


if __name__ == "__main__":
    main()
