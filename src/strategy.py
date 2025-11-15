from backtesting import Strategy
from backtesting.lib import crossover

from .indicators import sma


class SMACross(Strategy):
    fast_window: int = 20
    slow_window: int = 50

    def init(self):
        price = self.data.Close
        self.sma_fast = self.I(sma, price, self.fast_window)
        self.sma_slow = self.I(sma, price, self.slow_window)

    def next(self):
        if crossover(self.sma_fast, self.sma_slow):
            if not self.position.is_long:
                self.position.close()
                self.buy()
        elif crossover(self.sma_slow, self.sma_fast):
            if not self.position.is_short:
                self.position.close()
                self.sell()
