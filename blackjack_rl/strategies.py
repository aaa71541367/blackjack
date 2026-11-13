"""基準策略。

BasicStrategy：多副牌、S17、DAS、late surrender 的標準 basic strategy（下注固定最小注）。
在 H17 或其他規則下只是近似，但差異很小，足以當作驗證環境與比較 DQN 的基準。
"""
import numpy as np

from .config import DOUBLE, HIT, N_PLAY_ACTIONS, SPLIT, STICK, SURRENDER
from .env import BlackjackEnv, hand_value


def _up(d):          # 莊家明牌 2..11（A=11）
    return 11 if d == 1 else d


def basic_action(cards, dealer_up, mask) -> int:
    d = _up(dealer_up)
    v, soft = hand_value(cards)

    # 投降
    if mask[SURRENDER] and not soft and not (len(cards) == 2 and cards[0] == cards[1] == 8):
        if (v == 16 and d in (9, 10, 11)) or (v == 15 and d == 10):
            return SURRENDER

    # 分牌
    if mask[SPLIT]:
        c = cards[0]
        split = {
            1: True, 8: True, 10: False, 5: False,
            9: d in (2, 3, 4, 5, 6, 8, 9),
            7: d <= 7, 6: d <= 6, 4: d in (5, 6),
            3: d <= 7, 2: d <= 7,
        }[c]
        if split:
            return SPLIT

    def dbl_or(alt):
        return DOUBLE if mask[DOUBLE] else alt

    # 軟牌
    if soft:
        if v >= 19:
            return STICK
        if v == 18:
            if 3 <= d <= 6:
                return dbl_or(STICK)
            return STICK if d in (2, 7, 8) else HIT
        if v == 17:
            return dbl_or(HIT) if 3 <= d <= 6 else HIT
        if v in (15, 16):
            return dbl_or(HIT) if 4 <= d <= 6 else HIT
        if v in (13, 14):
            return dbl_or(HIT) if 5 <= d <= 6 else HIT
        return HIT

    # 硬牌
    if v >= 17:
        return STICK
    if 13 <= v <= 16:
        return STICK if d <= 6 else HIT
    if v == 12:
        return STICK if 4 <= d <= 6 else HIT
    if v == 11:
        return dbl_or(HIT) if d <= 10 else HIT
    if v == 10:
        return dbl_or(HIT) if d <= 9 else HIT
    if v == 9:
        return dbl_or(HIT) if 3 <= d <= 6 else HIT
    return HIT


class BasicStrategy:
    name = "basic_strategy"

    def act(self, env: BlackjackEnv, obs=None) -> int:
        mask = env.legal_mask()
        if env.phase == "bet":
            return N_PLAY_ACTIONS             # 永遠下最小注（bet_sizes[0]）
        h = env.hands[env.cur]
        a = basic_action(h.cards, env.dealer_cards[0], mask)
        if not mask[a]:                       # 例如分 A 後不能抽牌 → 停牌
            a = STICK
        return a


class RandomPolicy:
    name = "random"

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, env: BlackjackEnv, obs=None) -> int:
        return int(self.rng.choice(np.flatnonzero(env.legal_mask())))
