"""下注基準：以線性迴歸預測「這局以 1 倍注打完的期望報酬」，再依預測值決定下注。

下注是單步決策：下 b 倍的報酬正好是 1 倍的 b 倍（出牌不看下注），所以不需要 RL，
只要學一個「牌堆特徵 → 期望報酬」的函數。算牌研究指出這個關係近似線性
（每移除一張某點數的牌，對期望值的影響大致固定，即 effect of removal）。

特徵只用可見資訊，與 DQN 的 seen_counts 相同，不含副數或剩餘張數：
  [1, S, f_A, f_2, ..., f_7, f_9, f_10]
  S   = 已出現總張數 / 52
  f_i = (點數 i 已出現張數 − 已出現總數 × 該點數在整副牌中的比例) / 52
        即「這種牌比平均多出現了幾張」（以副為單位）
f_i 的總和恆為 0，所以以 8 點為參照不放入（文獻中 8 的影響接近 0），避免共線性。

用法：
  python -m blackjack_rl.bet_model --decks 4-8 --rounds 2000000 --out bet_model.npz
"""
import argparse
import time

import numpy as np

from .config import MAX_DECKS, N_PLAY_ACTIONS, RuleConfig, ObsConfig
from .env import BlackjackEnv
from .shoe import CARDS_PER_DECK

_SCALE = (CARDS_PER_DECK * MAX_DECKS).astype(np.float64)
CARD_NAMES = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10"]


_SHARE = CARDS_PER_DECK / 52.0
_KEEP = [i for i in range(10) if i != 7]       # 去掉 8 點（index 7）作為參照
N_FEATURES = 2 + len(_KEEP)


def features(seen: np.ndarray) -> np.ndarray:
    S = seen.sum()
    f = (seen - S * _SHARE) / 52.0
    return np.concatenate([[1.0, S / 52.0], f[_KEEP]])


def collect_bet_data(rules: RuleConfig, play_policy, n_rounds: int, seed: int = 0,
                     expected: bool = True, log=print):
    """收集下注訓練資料：每局在下注時記錄可見的已出現張數，固定下 1 倍注，
    以 play_policy 打完，目標為該局每單位下注的報酬（預設用期望結算降低雜訊）。
    回傳 (seen[n, 10], y[n])。"""
    rules = RuleConfig(**{**rules.__dict__, "enable_betting": True})
    env = BlackjackEnv(rules, ObsConfig("seen_counts"), seed=seed, expected_settlement=expected)
    seen = np.zeros((n_rounds, 10), np.int64)
    y = np.zeros(n_rounds)
    t0 = time.time()
    for k in range(n_rounds):
        obs, _ = env.reset()
        seen[k] = env.shoe.seen_counts
        obs, r, done, info = env.step(N_PLAY_ACTIONS)          # 固定下最小注
        while not done:
            obs, r, done, info = env.step(play_policy.act(env, obs))
        y[k] = info.get("expected_reward", r) / rules.bet_sizes[0]
        if log and (k + 1) % 250_000 == 0:
            log(f"  已收集 {k + 1:,} 局（{time.time() - t0:.0f}s）")
    return seen, y


def decide_bet(e: float, rules: RuleConfig, rule: str, thresholds) -> int:
    """依預測期望值 e 選下注倍數的 index（線性與神經網路下注共用）。"""
    n = len(rules.bet_sizes)
    if rule == "ev":
        return n - 1 if e > 0 else 0
    k = 0
    for j, th in enumerate(thresholds):
        if e > th:
            k = j + 1
    return min(k, n - 1)


class LinearBetModel:
    name = "linear_bet"

    def __init__(self, rules: RuleConfig, coef=None, rule: str = "ev", thresholds=None):
        """rule:
          "ev"     : 預測期望值 > 0 就下最大注，否則最小注（期望值最大化，不考慮風險）
          "spread" : 依 thresholds 分段加注，thresholds[k] 為使用第 k+1 個下注倍數的門檻
        """
        self.rules = rules
        self.coef = coef
        self.rule = rule
        n = len(rules.bet_sizes)
        self.thresholds = thresholds if thresholds is not None else \
            [0.0 + 0.005 * k for k in range(n - 1)]      # 預設：0%, 0.5%, 1% ...

    # ---------------------------------------------------------------- 訓練
    def fit(self, play_policy, n_rounds: int, seed: int = 0, expected: bool = True,
            ridge: float = 1e-6, log=print):
        seen, y = collect_bet_data(self.rules, play_policy, n_rounds, seed, expected, log)
        return self.fit_arrays(seen, y, ridge)

    def fit_arrays(self, seen: np.ndarray, y: np.ndarray, ridge: float = 1e-6):
        X = np.stack([features(s) for s in seen])
        A = X.T @ X + ridge * np.eye(X.shape[1])
        self.coef = np.linalg.solve(A, X.T @ y)
        resid = y - X @ self.coef
        self.r2 = 1 - resid.var() / y.var()
        return self

    # ---------------------------------------------------------------- 使用
    def predict(self, seen: np.ndarray) -> float:
        return float(features(seen) @ self.coef)

    def bet_index(self, seen: np.ndarray) -> int:
        return decide_bet(self.predict(seen), self.rules, self.rule, self.thresholds)

    def effect_of_removal(self) -> dict:
        """每多看到一張該點數的牌（相對於 8 點），期望值變化多少（%）。"""
        out = {}
        for j, i in enumerate(_KEEP):
            out[CARD_NAMES[i]] = 100 * self.coef[2 + j] / 52.0
        out["8"] = 0.0
        return {k: out[k] for k in CARD_NAMES}

    def save(self, path):
        np.savez(path, coef=self.coef, rule=self.rule, thresholds=np.array(self.thresholds))

    @classmethod
    def load(cls, path, rules: RuleConfig):
        d = np.load(path)
        return cls(rules, d["coef"], str(d["rule"]), list(d["thresholds"]))


class BaselinePolicy:
    """組合「出牌策略」與「下注模型」。bet_model=None 時固定最小注。
    出牌策略可以是 basic、DP 或只學出牌的 DQN；下注模型可以是線性或神經網路。"""

    def __init__(self, play_policy, bet_model=None, name=None):
        self.play, self.bet = play_policy, bet_model
        self.name = name or f"{play_policy.name}+{getattr(bet_model, 'name', 'min_bet')}"

    def act(self, env, obs=None) -> int:
        if env.phase == "bet":
            if self.bet is None:
                return N_PLAY_ACTIONS
            if hasattr(self.bet, "act_bet"):             # DQN 的下注頭
                return self.bet.act_bet(env, obs)
            return N_PLAY_ACTIONS + self.bet.bet_index(env.shoe.seen_counts)
        return self.play.act(env, obs)


def main():
    from .cli import add_rule_args, rules_from_args
    from .strategies import BasicStrategy

    p = argparse.ArgumentParser(description="訓練下注的線性迴歸模型")
    add_rule_args(p)
    p.add_argument("--rounds", type=int, default=2_000_000)
    p.add_argument("--rule", default="ev", choices=["ev", "spread"])
    p.add_argument("--no-expected", action="store_true", help="目標用實際結算（預設用期望結算）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="bet_model.npz")
    a = p.parse_args()
    rules = rules_from_args(a)
    m = LinearBetModel(rules, rule=a.rule).fit(BasicStrategy(), a.rounds, a.seed, not a.no_expected)
    m.save(a.out)
    print(f"R² = {m.r2:.4f}（單局結果雜訊很大，R² 很低是正常的）")
    print(f"剛洗好牌時的預測期望值：{m.coef[0] * 100:+.3f}%")
    print("每多看到一張該點數的牌（相對於 8 點），期望值變化（%）：")
    for k, v in m.effect_of_removal().items():
        print(f"  {k:>2}: {v:+.4f}")


if __name__ == "__main__":
    main()


class DQNBetHead:
    """把 DQN 的下注頭包成下注模型，可與任何出牌策略組合。"""
    name = "dqn_bet_head"

    def __init__(self, agent):
        self.agent = agent

    def act_bet(self, env, obs):
        return self.agent.act(env, obs)
