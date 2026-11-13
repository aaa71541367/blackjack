"""有限牌堆。

牌以點數表示：1 = A，2~9 = 數字牌，10 = 10/J/Q/K。
環境內部知道真實牌堆，但只有透過 reveal() 公開的牌會計入 seen_counts，
state 只會用到 seen_counts / running_count，不會用到剩餘張數或副數。
"""
import numpy as np

from .config import RuleConfig

CARDS_PER_DECK = np.array([4] * 9 + [16])   # index 0 = A ... index 9 = 10 點
HILO = {1: -1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: -1}


class Shoe:
    def __init__(self, rules: RuleConfig, rng: np.random.Generator):
        self.rules = rules
        self.rng = rng
        self.n_shuffles = 0
        self.n_emergency_shuffles = 0
        self.shuffle()

    # ------------------------------------------------------------------ 洗牌
    def shuffle(self):
        lo, hi = self.rules.deck_range
        self.n_decks = int(self.rng.integers(lo, hi + 1))
        counts = CARDS_PER_DECK * self.n_decks
        self.cards = np.repeat(np.arange(1, 11), counts)
        self.rng.shuffle(self.cards)
        self.pos = 0
        total = len(self.cards)
        pen = self.rng.uniform(*self.rules.penetration)
        self.cut = min(int(total * pen), total - self.rules.min_reserve)
        self.cut = max(self.cut, 1)
        self.seen_counts = np.zeros(10, dtype=np.int64)   # 玩家可見的已出現張數
        self.running_count = 0
        self.n_shuffles += 1
        for _ in range(self.rules.burn_cards):
            self.draw(visible=False)   # 燒牌不公開

    def needs_shuffle(self) -> bool:
        """只在局與局之間檢查：是否已經發過 cut card。"""
        return self.pos >= self.cut

    # ------------------------------------------------------------------ 發牌
    def draw(self, visible: bool = True) -> int:
        if self.pos >= len(self.cards):
            # 極少發生（單副牌 + 多次分牌）。視為換一副新牌堆，計數跟著歸零。
            self.n_emergency_shuffles += 1
            self.shuffle()
        card = int(self.cards[self.pos])
        self.pos += 1
        if visible:
            self.reveal(card)
        return card

    def reveal(self, card: int):
        """一張牌變成玩家看得到（例如莊家暗牌翻開）。"""
        self.seen_counts[card - 1] += 1
        self.running_count += HILO[card]

    # ------------------------------------------------------------------ 內部資訊（不可放入 state）
    @property
    def remaining(self) -> int:
        return len(self.cards) - self.pos

    # ------------------------------------------------------------------ 測試用
    def stack(self, next_cards):
        """把指定的牌移到接下來要發的位置（用交換，保持牌堆組成不變）。"""
        for i, want in enumerate(next_cards):
            p = self.pos + i
            idx = np.nonzero(self.cards[p:] == want)[0]
            if len(idx) == 0:
                raise ValueError(f"牌堆中已沒有點數 {want}")
            j = p + int(idx[0])
            self.cards[p], self.cards[j] = self.cards[j], self.cards[p]
