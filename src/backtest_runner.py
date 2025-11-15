from dataclasses import dataclass
from typing import Type, Dict, Any

import pandas as pd
from backtesting import Backtest, Strategy

from .utils import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    cash: float = 10000
    commission: float = 0.0005
    slippage: float = 0.0002


class BacktestRunner:
    def __init__(
        self,
        data: pd.DataFrame,
        strategy_cls: Type[Strategy],
        bt_config: BacktestConfig,
        strategy_params: Dict[str, Any] | None = None,
    ):
        self.data = data
        self.strategy_cls = strategy_cls
        self.bt_config = bt_config
        self.strategy_params = strategy_params or {}

    def run(self):
        logger.info(
            f"Running backtest with {self.strategy_cls.__name__} "
            f"params={self.strategy_params}"
        )

        bt = Backtest(
            self.data,
            self.strategy_cls,
            cash=self.bt_config.cash,
            commission=self.bt_config.commission,
            slippage=self.bt_config.slippage,
            exclusive_orders=True,
        )

        stats = bt.run(**self.strategy_params)
        return bt, stats
