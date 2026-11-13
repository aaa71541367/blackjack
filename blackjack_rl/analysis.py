"""訓練後分析。

1. strategy_table：在「剛洗好的牌堆」狀態下（DQN 與 tabular 皆適用），列出模型對每種起手牌 vs 莊家明牌的決策，
   並計算與 basic strategy 的一致率。
2. bet_by_true_count：評估時記錄每局下注，並依「環境內部」的 Hi-Lo true count 分組。
   true count 只用於事後分析，從未放入 state。
"""
import argparse
import unicodedata
from collections import defaultdict

import numpy as np

from .config import DOUBLE, N_PLAY_ACTIONS, SPLIT, SURRENDER, ObsConfig, RuleConfig
from .env import BlackjackEnv, hand_value
from .strategies import basic_action

SYM = {0: "S", 1: "H", 2: "R", 3: "P", 4: "D"}   # Stand Hit suRrender sPlit Double
UPS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 1]


def _pad(text, width):
    """考慮中文字佔兩格寬的左對齊。"""
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(0, width - w)


def _first_mask(rules: RuleConfig, cards):
    m = np.zeros(rules.n_actions, bool)
    m[0] = m[1] = True
    m[SURRENDER] = rules.late_surrender
    m[SPLIT] = cards[0] == cards[1] and rules.max_hands > 1
    v = hand_value(cards)[0]
    m[DOUBLE] = (rules.double_rule == "any" or (rules.double_rule == "9-11" and 9 <= v <= 11)
                 or (rules.double_rule == "10-11" and 10 <= v <= 11))
    return m


def _rows():
    hard = [(f"硬 {t}", [2, t - 2] if t <= 11 else [10, t - 10]) for t in range(5, 20)]
    soft = [(f"軟 {t} (A,{t - 11})", [1, t - 11]) for t in range(13, 21)]
    pairs = [(f"對 {'A' if c == 1 else c},{'A' if c == 1 else c}", [c, c]) for c in range(1, 11)]
    return [("硬牌", hard), ("軟牌", soft), ("對子", pairs)]


def strategy_table(agent, verbose=True):
    rules = agent.rules
    agree = total = 0
    out = []
    header = " " * 13 + "".join(f"{'A' if u == 1 else u:^5}" for u in UPS)
    for title, rows in _rows():
        out.append(f"\n[{title}]  模型決策（括號內為與 basic strategy 不同處的 basic 決策）")
        out.append(header)
        for label, cards in rows:
            m = _first_mask(rules, cards)
            cells = []
            for up in UPS:
                a = agent.first_decision_action(cards, up, m)
                b = basic_action(cards, up, m)
                total += 1
                agree += a == b
                cells.append(f"{SYM[a]:^5}" if a == b else f"{SYM[a]}({SYM[b]}) ")
            out.append(_pad(label, 13) + "".join(cells))
    rate = agree / total
    out.append("S=停牌 H=抽牌 R=投降 P=分牌 D=加倍")
    out.append(f"\n與 basic strategy 一致率：{rate:.1%}（{agree}/{total}）")
    if verbose:
        print("\n".join(out))
    return rate


def bet_by_true_count(agent, n_rounds=50_000, seed=7, verbose=True):
    rules, obs_cfg = agent.rules, agent.obs_cfg
    if not rules.enable_betting:
        return {}
    env = BlackjackEnv(rules, obs_cfg, seed=seed)
    buckets = defaultdict(lambda: [0, 0.0, 0.0])      # 局數、下注總和、報酬總和
    for _ in range(n_rounds):
        obs, _ = env.reset()
        decks_left = max(env.shoe.remaining / 52.0, 0.5)   # 分析用內部資訊
        tc = int(np.clip(np.floor(env.shoe.running_count / decks_left), -4, 6))
        a = agent.act(env, obs)
        bet = rules.bet_sizes[a - N_PLAY_ACTIONS]
        done = False
        while not done:
            obs, r, done, _ = env.step(a)
            if not done:
                a = agent.act(env, obs)
        b = buckets[tc]
        b[0] += 1; b[1] += bet; b[2] += r
    if verbose:
        print("\n依 Hi-Lo true count（僅分析用，不在 state 中）分組的下注行為：")
        print(" TC   局數比例   平均下注   每單位下注 EV")
        for tc in sorted(buckets):
            n, sb, sr = buckets[tc]
            print(f"{tc:>+3}   {n / n_rounds:7.1%}   {sb / n:8.2f}   {sr / sb * 100:+8.2f}%")
    return dict(buckets)


def load_any(path):
    """依副檔名載入：.pt → DQN，.pkl → tabular。"""
    if path.endswith(".pkl"):
        from .tabular import TabularAgent
        return TabularAgent.load(path)
    from .agent import DQNAgent
    return DQNAgent.load(path)


def main():
    p = argparse.ArgumentParser(description="分析訓練好的模型（.pt 或 .pkl）")
    p.add_argument("model")
    p.add_argument("--rounds", type=int, default=50_000)
    a = p.parse_args()
    agent = load_any(a.model)
    strategy_table(agent)
    bet_by_true_count(agent, a.rounds)


if __name__ == "__main__":
    main()
