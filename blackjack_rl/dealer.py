"""莊家最終結果的精確機率分布。

給定莊家目前的牌與牌堆「真實」剩餘組成，遞迴列舉莊家補牌的所有可能，
回傳 {結果: 機率}。結果為 17~21 的點數、"bust" 或 "bj"（歐式規則才會出現）。

這裡用到真實牌堆組成，但只用來計算「訓練目標」的期望值，不會進入 state：
對隱藏資訊取條件期望再平均，等於只用可見資訊的期望值，所以目標無偏、變異較小。
"""
from collections import defaultdict


def dealer_distribution(dealer_cards, counts, hit_soft17: bool, complete: bool = True) -> dict:
    """
    dealer_cards : 莊家目前的牌（1 或 2 張）
    counts       : 長度 10 的剩餘張數（index 0 = A ... index 9 = 10 點）
    complete     : False 表示莊家只補到兩張就停（所有手都爆牌或投降時，只需判斷是否 BJ）
    """
    memo = {}

    def rec(raw, has_ace, n, cnt):
        key = (raw, has_ace, n, cnt)
        if key in memo:
            return memo[key]
        soft = has_ace and raw + 10 <= 21
        v = raw + 10 if soft else raw
        res = None
        if n >= 2:
            if n == 2 and v == 21:
                res = {"bj": 1.0}
            elif v > 21:
                res = {"bust": 1.0}
            elif not complete:
                res = {v: 1.0}
            elif v > 17 or (v == 17 and not (soft and hit_soft17)):
                res = {v: 1.0}
        total = sum(cnt)
        if res is None and total == 0:           # 牌用完（極少見）：以目前點數結束
            res = {v if v <= 21 else "bust": 1.0}
        if res is None:
            acc = defaultdict(float)
            for i, c in enumerate(cnt):
                if c:
                    p = c / total
                    nxt = cnt[:i] + (c - 1,) + cnt[i + 1:]
                    for k, q in rec(raw + i + 1, has_ace or i == 0, n + 1, nxt).items():
                        acc[k] += p * q
            res = dict(acc)
        memo[key] = res
        return res

    raw = sum(dealer_cards)
    return rec(raw, 1 in dealer_cards, len(dealer_cards), tuple(int(c) for c in counts))


def dealer_distribution_fixed(dealer_cards, probs, hit_soft17: bool, complete: bool = True) -> dict:
    """快速近似版：莊家補牌時每張牌的機率固定為 probs（不因補牌而改變）。"""
    memo = {}

    def rec(raw, has_ace, n):
        key = (raw, has_ace, min(n, 3))
        if key in memo:
            return memo[key]
        soft = has_ace and raw + 10 <= 21
        v = raw + 10 if soft else raw
        res = None
        if n >= 2:
            if n == 2 and v == 21:
                res = {"bj": 1.0}
            elif v > 21:
                res = {"bust": 1.0}
            elif not complete:
                res = {v: 1.0}
            elif v > 17 or (v == 17 and not (soft and hit_soft17)):
                res = {v: 1.0}
        if res is None:
            acc = defaultdict(float)
            for i, p in enumerate(probs):
                if p > 0:
                    for k, q in rec(raw + i + 1, has_ace or i == 0, n + 1).items():
                        acc[k] += p * q
            res = dict(acc)
        memo[key] = res
        return res

    return rec(sum(dealer_cards), 1 in dealer_cards, len(dealer_cards))
