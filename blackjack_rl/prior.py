"""以 basic strategy 作為出牌的先驗。

basic strategy 的決策只取決於：點數、是否軟牌、對子、可否 double/split/surrender、莊家明牌，
這些都已經在 state 裡（見 env.encode_obs），所以可以預先建一張查表，
直接從 state 向量（torch tensor）批次查出 basic strategy 的動作。

BasicPrior(x) 回傳 (N, 5) 的先驗：basic strategy 的動作為 margin，其餘為 0；下注階段全為 0。
用途：
  - 殘差 Q 值：Q(s,a) = 網路(s,a) + BasicPrior(s,a)，網路輸出初始為 0 → 起點即 basic strategy
  - 模仿學習：提供專家動作
"""
import numpy as np
import torch
import torch.nn as nn

from .config import DOUBLE, N_PLAY_ACTIONS, SPLIT, SURRENDER, ObsConfig, RuleConfig
from .env import _TOTAL_MAX, _TOTAL_MIN
from .strategies import basic_action

N_TOT = _TOTAL_MAX - _TOTAL_MIN + 1        # 4..21 → 18


def _cards_for(total: int, soft: bool, pair: bool):
    """由可見特徵重建一組代表性的手牌（basic_action 只看點數、軟牌、對子點數）。"""
    if pair:
        c = 1 if (soft and total == 12) else total // 2
        return [c, c]
    if soft:
        return [1, total - 11]
    if total >= 12:
        return [10, total - 10]
    a = 2 if total - 2 != 2 else 3          # 避開對子
    return [a, total - a]


def build_table(rules: RuleConfig) -> np.ndarray:
    """table[total, soft, pair, can_d, can_s, can_r, dealer] = basic strategy 動作。"""
    tab = np.zeros((N_TOT, 2, 2, 2, 2, 2, 10), dtype=np.int64)
    for ti in range(N_TOT):
        total = ti + _TOTAL_MIN
        for soft in (0, 1):
            for pair in (0, 1):
                valid = True
                if soft and total < 12:
                    valid = False
                if pair and (total % 2 == 1 and not (soft and total == 12)):
                    valid = False
                cards = _cards_for(total, bool(soft), bool(pair)) if valid else [10, max(total - 10, 2)]
                for cd in (0, 1):
                    for cs in (0, 1):
                        for cr in (0, 1):
                            m = np.zeros(N_PLAY_ACTIONS, bool)
                            m[0] = m[1] = True
                            m[DOUBLE], m[SPLIT], m[SURRENDER] = bool(cd), bool(cs and pair), bool(cr)
                            for up in range(1, 11):
                                try:
                                    a = basic_action(cards, up, m)
                                except (KeyError, IndexError):
                                    a = 0
                                tab[ti, soft, pair, cd, cs, cr, up - 1] = a
    return tab


class BasicPrior(nn.Module):
    def __init__(self, rules: RuleConfig, obs_cfg: ObsConfig, margin: float = 0.02):
        super().__init__()
        self.onehot = obs_cfg.total_encoding == "onehot"
        self.j = 1 + N_TOT if self.onehot else 2      # 軟牌特徵的位置（見 encode_obs 版面）
        self.margin = margin
        self.register_buffer("table", torch.as_tensor(build_table(rules)))

    @torch.no_grad()
    def actions(self, x: torch.Tensor) -> torch.Tensor:
        """回傳每列的 basic strategy 動作；下注階段回傳 -1。"""
        if self.onehot:
            ti = x[:, 1:1 + N_TOT].argmax(1)
        else:
            ti = (x[:, 1] * 21).round().long().clamp(_TOTAL_MIN, _TOTAL_MAX) - _TOTAL_MIN
        j = self.j
        b = lambda k: (x[:, j + k] > 0.5).long()
        soft, pair, cd, cs, cr = b(0), b(1), b(5), b(6), b(7)
        up = x[:, j + 8:j + 18].argmax(1)
        a = self.table[ti, soft, pair, cd, cs, cr, up]
        return torch.where(x[:, 0] > 0.5, torch.full_like(a, -1), a)

    @torch.no_grad()
    def legal_from_obs(self, x: torch.Tensor) -> torch.Tensor:
        """由 state 推出出牌階段的合法動作（停牌、抽牌恆為合法；分 A 後不能抽牌的少數情況無法
        從 state 判斷，視為可抽牌，只影響間距損失中取最大值的範圍）。"""
        j = self.j
        m = torch.ones(len(x), N_PLAY_ACTIONS, dtype=torch.bool, device=x.device)
        m[:, SURRENDER] = x[:, j + 7] > 0.5
        m[:, SPLIT] = x[:, j + 6] > 0.5
        m[:, DOUBLE] = x[:, j + 5] > 0.5
        return m

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.actions(x)
        out = torch.zeros(len(x), N_PLAY_ACTIONS, device=x.device)
        rows = torch.nonzero(a >= 0).squeeze(1)
        out[rows, a[rows]] = self.margin
        return out
