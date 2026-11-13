"""只用可見資訊的出牌最佳化（DP 基準，作為 DQN 的近似上限）。

可用資訊與 DQN 相同：自己的牌、莊家明牌、玩家看得到的各點數已出現張數（seen_counts），
以及規則設定（副數的可能範圍、cut card 範圍）。完全不讀取真實牌堆、實際副數或暗牌。

步驟：
  1. 副數後驗：P(n | 已出現) ∝ 先驗(n) × 多變量超幾何機率 × P(尚未發到 cut card | n)
  2. 每種副數 n 下，未見牌池 = n 副的總張數 − 已出現。暗牌、燒牌都視為未見牌池中的牌
     （對玩家而言它們和尚未發出的牌可互換）。美式規則下以「莊家不是 BJ」為條件
  3. 精確計算停牌、抽牌、加倍、投降的期望值（抽牌會把抽到的牌從牌池移除）
  4. 依後驗加權，選期望值最大的合法動作

兩種模式：
  - exact=True ：莊家與玩家補牌時都把抽到的牌從牌池移除（較慢）
  - exact=False：一手牌進行中，每張牌的機率固定為決策當下的牌池比例（快很多，預設）

近似之處（組成相依策略計算器的常見做法，誤差通常很小）：
  - 莊家結果的機率分布以決策當下的牌池計算，不隨玩家之後抽到的牌更新
  - 分牌以「兩手獨立、不再分牌」估計：2 × 單手期望值
"""
from functools import lru_cache
from math import lgamma

import numpy as np

from .config import DOUBLE, HIT, N_PLAY_ACTIONS, SPLIT, STICK, SURRENDER, RuleConfig
from .dealer import dealer_distribution, dealer_distribution_fixed
from .env import BlackjackEnv, hand_value
from .shoe import CARDS_PER_DECK

_LOG_PRIOR_FLOOR = 1e-12


def _log_comb(n, k):
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def deck_posterior(rules: RuleConfig, seen: np.ndarray, dealt_before_round: int) -> dict:
    """回傳 {副數: 後驗機率}。

    seen               : 各點數已出現張數（可見）
    dealt_before_round : 本局開始前已發出的張數（燒牌 + 之前各局可見的牌），用來判斷
                         「還沒發到 cut card」對副數的限制
    """
    lo, hi = rules.deck_range
    S = int(seen.sum())
    p_lo, p_hi = rules.penetration
    logs = {}
    for n in range(lo, hi + 1):
        K = CARDS_PER_DECK * n
        T = int(K.sum())
        if np.any(seen > K):
            continue
        # 多變量超幾何：已出現的牌是從 n 副牌中隨機抽出
        ll = sum(_log_comb(int(K[i]), int(seen[i])) for i in range(10)) - _log_comb(T, S)
        # 本局開始時尚未發到 cut card 的機率（cut 位置在 p_lo~p_hi 間均勻）
        x = dealt_before_round
        if x >= T - rules.min_reserve:
            continue
        surv = np.clip((p_hi - x / T) / max(p_hi - p_lo, 1e-9), 0.0, 1.0) if p_hi > p_lo \
            else float(x < p_lo * T)
        if surv <= 0:
            continue
        logs[n] = ll + np.log(max(surv, _LOG_PRIOR_FLOOR))
    if not logs:                          # 理論上不會發生；保底用均勻先驗
        return {n: 1.0 / (hi - lo + 1) for n in range(lo, hi + 1)}
    m = max(logs.values())
    w = {n: np.exp(v - m) for n, v in logs.items()}
    z = sum(w.values())
    return {n: v / z for n, v in w.items()}


class _Evaluator:
    """固定一個未見牌池與莊家明牌，計算各動作的期望值（以 1 倍注為單位）。"""

    def __init__(self, rules: RuleConfig, pool: np.ndarray, up: int, exact: bool = False):
        self.r = rules
        self.exact = exact
        self.pool0 = tuple(int(c) for c in pool)
        if exact:
            dist = dealer_distribution([up], pool, rules.dealer_hits_soft17, complete=True)
        else:
            probs = np.asarray(pool, float) / max(1, sum(self.pool0))
            dist = dealer_distribution_fixed([up], probs, rules.dealer_hits_soft17, complete=True)
        if rules.dealer_peek:
            # 美式：莊家已 peek 確認不是 BJ，對其餘結果重新正規化
            p_nobj = 1.0 - dist.get("bj", 0.0)
            dist = {k: v / p_nobj for k, v in dist.items() if k != "bj"}
            self.p_bj = 0.0
        else:
            self.p_bj = dist.get("bj", 0.0)
        self.dist = dist
        # 停牌期望值（依玩家點數）：bj 計為輸
        self.stand = {}
        for t in range(4, 22):
            e = 0.0
            for k, p in dist.items():
                if k == "bj":
                    e -= p
                elif k == "bust" or t > k:
                    e += p
                elif t < k:
                    e -= p
            self.stand[t] = e
        self._hit_memo = {}

    def stand_ev(self, cards_raw, has_ace):
        v = cards_raw + 10 if has_ace and cards_raw + 10 <= 21 else cards_raw
        return -1.0 if v > 21 else self.stand[max(v, 4)]

    def best_after_hit(self, raw, ace, pool):
        """已經抽過牌的手：只能停牌或繼續抽牌，回傳最佳期望值。"""
        key = (raw, ace, pool)
        if key in self._hit_memo:
            return self._hit_memo[key]
        v = raw + 10 if ace and raw + 10 <= 21 else raw
        if v > 21:
            res = -1.0
        elif v == 21:                     # 21 點自動停牌
            res = self.stand[21]
        else:
            res = max(self.stand[max(v, 4)], self.hit_ev(raw, ace, pool))
        self._hit_memo[key] = res
        return res

    def hit_ev(self, raw, ace, pool):
        total = sum(pool)
        if total == 0:
            return self.stand_ev(raw, ace)
        e = 0.0
        for i, c in enumerate(pool):
            if c:
                nxt = pool[:i] + (c - 1,) + pool[i + 1:] if self.exact else pool
                e += c / total * self.best_after_hit(raw + i + 1, ace or i == 0, nxt)
        return e

    def double_ev(self, raw, ace, pool):
        total = sum(pool)
        e = 0.0
        for i, c in enumerate(pool):
            if c:
                e += c / total * self.stand_ev(raw + i + 1, ace or i == 0)
        return 2.0 * e

    def surrender_ev(self):
        return -0.5 * (1 - self.p_bj) - self.p_bj

    def split_ev(self, card):
        """近似：2 × 單手期望值（單手 = 一張 card 再補一張，之後最佳化，不再分牌）。"""
        r, pool = self.r, self.pool0
        total = sum(pool)
        e = 0.0
        for i, c in enumerate(pool):
            if not c:
                continue
            nxt = pool[:i] + (c - 1,) + pool[i + 1:] if self.exact else pool
            raw, ace = card + i + 1, card == 1 or i == 0
            v = raw + 10 if ace and raw + 10 <= 21 else raw
            if card == 1 and not r.hit_split_aces:
                best = self.stand_ev(raw, ace)
            elif v == 21:
                best = self.stand[21]         # 分牌後的 21 點不是 BJ，照一般 21 結算
            else:
                opts = [self.stand_ev(raw, ace), self.hit_ev(raw, ace, nxt)]
                if r.double_after_split and self._double_allowed(v):
                    opts.append(self.double_ev(raw, ace, nxt))
                best = max(opts)
            e += c / total * best
        return 2.0 * e

    def _double_allowed(self, v):
        rule = self.r.double_rule
        return rule == "any" or (rule == "9-11" and 9 <= v <= 11) or (rule == "10-11" and 10 <= v <= 11)

    def action_evs(self, cards, mask):
        raw, ace = sum(cards), 1 in cards
        pool = self.pool0
        ev = np.full(len(mask), -np.inf)
        ev[STICK] = self.stand_ev(raw, ace)
        if mask[HIT]:
            ev[HIT] = self.hit_ev(raw, ace, pool)
        if mask[DOUBLE]:
            ev[DOUBLE] = self.double_ev(raw, ace, pool)
        if mask[SURRENDER]:
            ev[SURRENDER] = self.surrender_ev()
        if mask[SPLIT]:
            ev[SPLIT] = self.split_ev(cards[0])
        return ev


def visible_info(env: BlackjackEnv):
    """從環境取出「玩家看得到」的資訊。這是 DP 基準唯一的資料來源。"""
    h = env.hands[env.cur]
    seen = env.shoe.seen_counts.copy()
    # 本局可見的牌：所有手牌 + 莊家明牌（暗牌尚未翻開）
    this_round = sum(len(x.cards) for x in env.hands) + 1
    dealt_before = env.rules.burn_cards + int(seen.sum()) - this_round
    return h.cards, env.dealer_cards[0], env.legal_mask(), seen, dealt_before


def optimal_action_evs(rules: RuleConfig, cards, up, mask, seen, dealt_before,
                       min_weight=1e-4, exact=False):
    """依副數後驗加權的各動作期望值。"""
    post = deck_posterior(rules, seen, dealt_before)
    ev = np.zeros(len(mask))
    wsum = 0.0
    for n, w in post.items():
        if w < min_weight:
            continue
        pool = CARDS_PER_DECK * n - seen
        ev += w * _Evaluator(rules, pool, up, exact).action_evs(cards, mask)
        wsum += w
    return ev / wsum


class OptimalPlay:
    """DP 出牌基準。下注階段固定下最小注（下注由 bet_model 另外處理）。"""
    name = "dp_play"

    def __init__(self, rules: RuleConfig, exact: bool = False):
        self.rules = rules
        self.exact = exact
        self.name = "dp_play_exact" if exact else "dp_play"

    def first_decision_action(self, cards, dealer_up, mask) -> int:
        """剛洗好牌時的起手決策（策略表用）：可見的只有自己的兩張牌與莊家明牌。"""
        seen = np.zeros(10, dtype=np.int64)
        for c in list(cards) + [dealer_up]:
            seen[c - 1] += 1
        ev = optimal_action_evs(self.rules, cards, dealer_up, mask, seen, self.rules.burn_cards,
                                exact=self.exact)
        return int(np.where(mask[:len(ev)], ev, -np.inf).argmax())

    def act(self, env: BlackjackEnv, obs=None) -> int:
        if env.phase == "bet":
            return N_PLAY_ACTIONS
        cards, up, mask, seen, dealt_before = visible_info(env)
        ev = optimal_action_evs(env.rules, cards, up, mask, seen, dealt_before, exact=self.exact)
        ev = np.where(mask, ev, -np.inf)
        return int(ev.argmax())
