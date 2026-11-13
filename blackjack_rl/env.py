"""Blackjack 環境。

一局（round）= 一個 episode：
  [下注決策] -> 發牌 -> [玩家決策 ...] -> 莊家補牌 -> 結算（done=True）
牌堆在局與局之間延續，只在發過 cut card 後的下一局開始前重洗。

reward 以「基本注 1」為單位：輸 -bet、平手 0、贏 +bet、Blackjack +payout*bet、投降 -0.5*bet。
所有中間步驟 reward 為 0，只有結束該局的那一步有 reward。
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .config import (DOUBLE, HIT, MAX_DECKS, N_PLAY_ACTIONS, SPLIT, STICK,
                     SURRENDER, ObsConfig, RuleConfig)
from .dealer import dealer_distribution
from .shoe import CARDS_PER_DECK, Shoe


# ---------------------------------------------------------------------- 手牌計算
def hand_value(cards) -> tuple:
    """回傳 (最佳點數, 是否軟牌)。A 最多只有一張能算 11。"""
    total = sum(cards)
    if 1 in cards and total + 10 <= 21:
        return total + 10, True
    return total, False


@dataclass
class Hand:
    cards: List[int]
    bet: float
    is_split: bool = False
    from_aces: bool = False
    done: bool = False
    surrendered: bool = False
    doubled: bool = False

    @property
    def value(self) -> int:
        return hand_value(self.cards)[0]

    @property
    def soft(self) -> bool:
        return hand_value(self.cards)[1]

    @property
    def bust(self) -> bool:
        return self.value > 21


# ---------------------------------------------------------------------- 觀測編碼
_SEEN_SCALE = (CARDS_PER_DECK * MAX_DECKS).astype(np.float32)   # 只用來縮放，與實際副數無關
_HILO_SCALE = 4.0 * MAX_DECKS
_TOTAL_MIN, _TOTAL_MAX = 4, 21                                     # one-hot 的點數範圍


def play_feature_dim(obs_cfg: ObsConfig) -> int:
    total_dim = _TOTAL_MAX - _TOTAL_MIN + 1 if obs_cfg.total_encoding == "onehot" else 1
    return 9 + total_dim + 10


def shoe_feature_dim(obs_cfg: ObsConfig) -> int:
    return {"none": 0, "seen_counts": 10, "hilo": 1}[obs_cfg.shoe_info]


def encode_obs(rules: RuleConfig, obs_cfg: ObsConfig, *, phase: str,
               hand_cards=None, is_split=False, n_hands=1, dealer_up=None,
               mask=None, seen_counts=None, running_count=0) -> np.ndarray:
    """把遊戲狀態轉成向量。刻意不接受副數或剩餘張數作為輸入。

    版面：[下注階段, 點數(1 或 18 維), 軟牌, 對子, 第一個決策, 分牌後的手, 手數,
           可 double, 可 split, 可投降, 莊家明牌 one-hot(10), 牌堆特徵]
    """
    pdim = play_feature_dim(obs_cfg)
    x = np.zeros(pdim + shoe_feature_dim(obs_cfg), dtype=np.float32)
    if phase == "bet":
        x[0] = 1.0
    elif phase == "play":
        v, soft = hand_value(hand_cards)
        first = len(hand_cards) == 2
        if obs_cfg.total_encoding == "onehot":
            x[1 + int(np.clip(v, _TOTAL_MIN, _TOTAL_MAX)) - _TOTAL_MIN] = 1.0
            j = 1 + _TOTAL_MAX - _TOTAL_MIN + 1
        else:
            x[1] = v / 21.0
            j = 2
        x[j] = float(soft)
        x[j + 1] = float(first and hand_cards[0] == hand_cards[1])
        x[j + 2] = float(first)
        x[j + 3] = float(is_split)
        x[j + 4] = n_hands / rules.max_hands
        x[j + 5] = float(mask[DOUBLE])
        x[j + 6] = float(mask[SPLIT])
        x[j + 7] = float(mask[SURRENDER])
        x[j + 8 + dealer_up - 1] = 1.0            # 莊家明牌 one-hot（A 在第一格）
    if obs_cfg.shoe_info == "seen_counts":
        sc = np.zeros(10) if seen_counts is None else seen_counts
        x[pdim:] = sc / _SEEN_SCALE
    elif obs_cfg.shoe_info == "hilo":
        x[pdim] = running_count / _HILO_SCALE
    return x


# ---------------------------------------------------------------------- 環境
class BlackjackEnv:
    def __init__(self, rules: Optional[RuleConfig] = None,
                 obs_cfg: Optional[ObsConfig] = None, seed: Optional[int] = None,
                 expected_settlement: bool = False):
        """expected_settlement=True 時，局結束的 info 會多一個 "expected_reward"：
        以牌堆真實組成計算的期望報酬，只供訓練目標使用（評估請用實際 reward）。"""
        self.expected_settlement = expected_settlement
        self._pending_expected = None
        self.rules = rules or RuleConfig()
        self.obs_cfg = obs_cfg or ObsConfig()
        self.rng = np.random.default_rng(seed)
        self.shoe = Shoe(self.rules, self.rng)
        self.n_actions = self.rules.n_actions
        self.obs_dim = play_feature_dim(self.obs_cfg) + shoe_feature_dim(self.obs_cfg)
        self.phase = "done"
        self.hands: List[Hand] = []
        self.cur = 0
        self.dealer_cards: List[int] = []
        self.hole: Optional[int] = None
        self.base_bet = 1.0

    # ------------------------------------------------------------------ 公開介面
    def reset(self):
        """開始新的一局。

        開啟下注決策時，一定回傳下注階段的 state。
        關閉下注決策時，玩家無需動作的局（莊家 peek 到 BJ、玩家天生 BJ）會自動結算，
        其報酬放在 info["auto_rewards"]，評估時必須一併計入。
        """
        auto = []
        while True:
            self._start_round()
            if self.rules.enable_betting:
                self.phase = "bet"
                break
            self.base_bet = 1.0
            r = self._deal()
            if r is None:
                self.phase = "play"
                break
            auto.append(r)
        return self.observe(), {"auto_rewards": auto}

    def step(self, action: int):
        action = int(action)
        assert 0 <= action < self.n_actions, f"invalid action {action}"
        assert self.phase in ("bet", "play"), "episode 已結束，請先 reset()"
        assert self.legal_mask()[action], f"illegal action {action} (phase={self.phase})"

        if self.phase == "bet":
            self.base_bet = float(self.rules.bet_sizes[action - N_PLAY_ACTIONS])
            r = self._deal()
            if r is not None:
                return self._end_round(r)
            self.phase = "play"
            return self._continue()

        h = self.hands[self.cur]
        if action == STICK:
            h.done = True
        elif action == HIT:
            h.cards.append(self.shoe.draw())
            if h.value >= 21:             # 爆牌或 21 點自動停牌
                h.done = True
        elif action == SURRENDER:
            h.surrendered = True
            h.done = True
        elif action == SPLIT:
            c0, c1 = h.cards
            aces = c0 == 1
            h.cards, h.is_split, h.from_aces = [c0], True, aces
            new = Hand([c1], bet=h.bet, is_split=True, from_aces=aces)
            self.hands.insert(self.cur + 1, new)
            h.cards.append(self.shoe.draw())   # 第二手的第二張牌等輪到它才發
            self._after_split_card(h)
        elif action == DOUBLE:
            h.bet *= 2
            h.doubled = True
            h.cards.append(self.shoe.draw())
            h.done = True
        return self._advance()

    def legal_mask(self) -> np.ndarray:
        r = self.rules
        m = np.zeros(self.n_actions, dtype=bool)
        if self.phase == "bet":
            m[N_PLAY_ACTIONS:] = True
            return m
        if self.phase != "play":
            return m
        h = self.hands[self.cur]
        first = len(h.cards) == 2
        ace_locked = h.from_aces and not r.hit_split_aces
        m[STICK] = True
        m[HIT] = not ace_locked
        m[SURRENDER] = r.late_surrender and first and len(self.hands) == 1
        if first and h.cards[0] == h.cards[1] and len(self.hands) < r.max_hands:
            m[SPLIT] = not (h.cards[0] == 1 and h.is_split and not r.resplit_aces)
        if first and not ace_locked and (not h.is_split or r.double_after_split):
            v = h.value
            m[DOUBLE] = (r.double_rule == "any"
                         or (r.double_rule == "9-11" and 9 <= v <= 11)
                         or (r.double_rule == "10-11" and 10 <= v <= 11))
        return m

    def observe(self) -> np.ndarray:
        kw = dict(seen_counts=self.shoe.seen_counts, running_count=self.shoe.running_count)
        if self.phase == "play":
            h = self.hands[self.cur]
            return encode_obs(self.rules, self.obs_cfg, phase="play", hand_cards=h.cards,
                              is_split=h.is_split, n_hands=len(self.hands),
                              dealer_up=self.dealer_cards[0], mask=self.legal_mask(), **kw)
        return encode_obs(self.rules, self.obs_cfg, phase=self.phase, **kw)

    # ------------------------------------------------------------------ 內部流程
    def _start_round(self):
        if self.shoe.needs_shuffle():
            self.shoe.shuffle()
        self.hands, self.cur = [], 0
        self.dealer_cards, self.hole = [], None

    def _deal(self) -> Optional[float]:
        """發起手牌並處理 Blackjack。若這局不需要玩家決策就結束，回傳 reward；否則回傳 None。"""
        s, r, bet = self.shoe, self.rules, self.base_bet
        p1, up, p2 = s.draw(), s.draw(), s.draw()
        self.hands = [Hand([p1, p2], bet=bet)]
        self.dealer_cards = [up]
        player_bj = hand_value([p1, p2])[0] == 21

        if r.dealer_peek:
            self.hole = s.draw(visible=False)       # 暗牌：已離開牌堆，但玩家看不到
            dealer_bj = hand_value([up, self.hole])[0] == 21
            if dealer_bj or player_bj:
                self._reveal_hole()
                if dealer_bj:
                    return 0.0 if player_bj else -bet
                return r.blackjack_payout * bet
        elif player_bj:                              # 歐式：玩家 BJ 時才補莊家第二張
            if self.expected_settlement:
                dist = dealer_distribution([up], self._remaining_counts(), r.dealer_hits_soft17, False)
                self._pending_expected = (1 - dist.get("bj", 0.0)) * r.blackjack_payout * bet
            self.dealer_cards.append(s.draw())
            return 0.0 if hand_value(self.dealer_cards)[0] == 21 else r.blackjack_payout * bet
        return None

    def _reveal_hole(self):
        if self.hole is not None:
            self.shoe.reveal(self.hole)
            self.dealer_cards.append(self.hole)
            self.hole = None

    def _after_split_card(self, h: Hand):
        """分牌後的手拿到第二張牌時的處理。"""
        if h.from_aces and not self.rules.hit_split_aces:
            can_resplit = (self.rules.resplit_aces and h.cards == [1, 1]
                           and len(self.hands) < self.rules.max_hands)
            if not can_resplit:
                h.done = True
        if h.value == 21:
            h.done = True

    def _advance(self):
        while self.cur < len(self.hands):
            h = self.hands[self.cur]
            if len(h.cards) == 1:                    # 輪到分牌後的下一手，這時才發第二張
                h.cards.append(self.shoe.draw())
                self._after_split_card(h)
            if not h.done:
                return self._continue()
            self.cur += 1
        return self._end_round(self._dealer_and_settle())

    def _continue(self):
        return self.observe(), 0.0, False, {}

    def _end_round(self, reward: float):
        self.phase = "done"
        info = {"round_reward": reward, "base_bet": self.base_bet,
                "n_hands": len(self.hands)}
        if self.expected_settlement:
            e = self._pending_expected
            info["expected_reward"] = float(reward if e is None else e)
            self._pending_expected = None
        return self.observe(), float(reward), True, info

    def _remaining_counts(self):
        """牌堆真實剩餘組成（含暗牌以外尚未發出的牌）。只用於期望結算，絕不放入 state。"""
        return np.bincount(self.shoe.cards[self.shoe.pos:], minlength=11)[1:]

    def _settle(self, dv, dealer_bj: bool) -> float:
        """給定莊家最終點數（爆牌以 22 表示）與是否 BJ，計算所有手的總報酬。"""
        total = 0.0
        for h in self.hands:
            if h.surrendered:
                total += -h.bet if dealer_bj else -0.5 * h.bet
            elif h.bust or dealer_bj:                # 歐式莊家 BJ：加倍與分牌的注全輸
                total -= h.bet
            elif dv > 21 or h.value > dv:
                total += h.bet
            elif h.value < dv:
                total -= h.bet
        return total

    def _expected_from_dist(self, dist) -> float:
        e = 0.0
        for k, p in dist.items():
            if k == "bj":
                e += p * self._settle(21, True)
            elif k == "bust":
                e += p * self._settle(22, False)
            else:
                e += p * self._settle(k, False)
        return e

    def _dealer_and_settle(self) -> float:
        r, s = self.rules, self.shoe
        live = any(not h.bust and not h.surrendered for h in self.hands)
        any_surr = any(h.surrendered for h in self.hands)

        if self.expected_settlement:                 # 在莊家動作之前計算期望值
            start = [self.dealer_cards[0]] + ([self.hole] if self.hole is not None else [])
            if live or (any_surr and not r.dealer_peek):
                dist = dealer_distribution(start, self._remaining_counts(),
                                           r.dealer_hits_soft17, complete=live)
                self._pending_expected = self._expected_from_dist(dist)
            # 否則結果已確定（全部爆牌/投降且莊家不會 BJ），用實際報酬即可

        if r.dealer_peek:
            self._reveal_hole()                      # 莊家一定會翻開暗牌
        elif live or any_surr:
            self.dealer_cards.append(s.draw())       # 歐式：此時才發第二張

        if live:                                     # 所有手都爆牌/投降時莊家不補牌
            while True:
                v, soft = hand_value(self.dealer_cards)
                if v < 17 or (v == 17 and soft and r.dealer_hits_soft17):
                    self.dealer_cards.append(s.draw())
                else:
                    break

        dv = hand_value(self.dealer_cards)[0]
        dealer_bj = len(self.dealer_cards) == 2 and dv == 21   # 只會在歐式發生
        return self._settle(dv, dealer_bj)
