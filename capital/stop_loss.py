from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StopLossTracker:
    """跟踪资金状态并在触发止损条件时暂停投注。

    三重止损规则（来自设计文档）：
    1. 单日亏损上限：当日拟投注总额 > 当前资本 × daily_loss_limit_pct → 拒绝投注。
    2. 连亏天数上限：连续亏损天数 >= max_consecutive_loss_days → 暂停。
    3. 总回撤上限：(峰值资本 - 当前资本) / 峰值资本 >= max_drawdown_pct → 暂停。

    典型参数（设计文档默认值）：
        daily_loss_limit_pct = 0.10（单日最多亏损资本的 10%）
        max_consecutive_loss_days = 3（连亏 3 天暂停）
        max_drawdown_pct = 0.30（总回撤 30% 暂停）
    """

    initial_capital: float
    daily_loss_limit_pct: float = 0.10
    max_consecutive_loss_days: int = 3
    max_drawdown_pct: float = 0.30

    _current_capital: float = field(init=False, repr=False)
    _peak_capital: float = field(init=False, repr=False)
    _consecutive_loss_days: int = field(init=False, default=0, repr=False)
    _paused: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital 必须大于 0")
        self._current_capital = self.initial_capital
        self._peak_capital = self.initial_capital

    # ──────────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────────

    def should_bet(self, proposed_daily_stake: float) -> bool:
        """判断当日是否允许下注。

        Args:
            proposed_daily_stake: 当日拟下注的总金额。

        Returns:
            True 表示允许投注，False 表示触发止损。
        """
        if self._paused:
            return False
        # 总回撤实时检查（可能在上一天已接近边界）
        if self._drawdown() >= self.max_drawdown_pct:
            self._paused = True
            return False
        # 单日亏损上限
        return proposed_daily_stake <= self._current_capital * self.daily_loss_limit_pct

    def record_day(self, daily_profit: float) -> None:
        """记录一天的盈亏，并更新止损状态。

        Args:
            daily_profit: 当日净盈亏（负数为亏损）。
        """
        self._current_capital += daily_profit
        if self._current_capital > self._peak_capital:
            self._peak_capital = self._current_capital

        if daily_profit < 0:
            self._consecutive_loss_days += 1
        else:
            self._consecutive_loss_days = 0

        if self._consecutive_loss_days >= self.max_consecutive_loss_days:
            self._paused = True
        if self._drawdown() >= self.max_drawdown_pct:
            self._paused = True

    def resume(self) -> None:
        """人工确认后恢复投注（重置连亏计数）。"""
        self._paused = False
        self._consecutive_loss_days = 0

    # ──────────────────────────────────────────────
    # 只读属性
    # ──────────────────────────────────────────────

    @property
    def capital(self) -> float:
        return self._current_capital

    @property
    def peak_capital(self) -> float:
        return self._peak_capital

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def consecutive_loss_days(self) -> int:
        return self._consecutive_loss_days

    @property
    def drawdown(self) -> float:
        return self._drawdown()

    # ──────────────────────────────────────────────
    # 内部辅助
    # ──────────────────────────────────────────────

    def _drawdown(self) -> float:
        if self._peak_capital == 0:
            return 0.0
        return (self._peak_capital - self._current_capital) / self._peak_capital
