"""Tabular 對照組：Monte Carlo control 與 Q-learning。

無牌堆資訊時 state 空間很小（約數千個狀態），tabular 方法可以逼近最佳策略，
作為 DQN 的參考上限：若 tabular 收斂而 DQN 不收斂，問題在網路訓練而非環境。

state key 只由玩家看得到的資訊組成，和 DQN 的 state 遵守同樣的資訊限制：
  - "none" : 牌局資訊
  - "hilo" : 牌局資訊 + Hi-Lo running count（截斷在 ±rc_clip）
  - "seen_counts" 是 10 維計數，組合數過大，不支援 tabular
"""
import argparse
import os
import pickle
import time
from dataclasses import asdict

import numpy as np

from .config import DOUBLE, SPLIT, SURRENDER, ObsConfig, RuleConfig
from .env import BlackjackEnv, hand_value


def table_key(obs_cfg: ObsConfig, phase: str, hand_cards=None, is_split=False,
              dealer_up=None, mask=None, running_count=0, rc_clip=10) -> tuple:
    if obs_cfg.shoe_info == "seen_counts":
        raise ValueError("tabular 不支援 seen_counts，請用 none 或 hilo")
    rc = (int(np.clip(running_count, -rc_clip, rc_clip)),) if obs_cfg.shoe_info == "hilo" else ()
    if phase == "bet":
        return ("bet",) + rc
    v, soft = hand_value(hand_cards)
    first = len(hand_cards) == 2
    pair = hand_cards[0] if first and hand_cards[0] == hand_cards[1] else 0
    # 手數只影響能否再分牌，已包含在 mask[SPLIT] 中
    return (v, soft, pair, first, is_split, bool(mask[DOUBLE]), bool(mask[SPLIT]),
            bool(mask[SURRENDER]), dealer_up) + rc


def env_key(env: BlackjackEnv, rc_clip=10) -> tuple:
    rc = env.shoe.running_count
    if env.phase == "bet":
        return table_key(env.obs_cfg, "bet", running_count=rc, rc_clip=rc_clip)
    h = env.hands[env.cur]
    return table_key(env.obs_cfg, "play", h.cards, h.is_split, env.dealer_cards[0],
                     env.legal_mask(), rc, rc_clip)


class TabularAgent:
    name = "tabular"

    def __init__(self, rules: RuleConfig, obs_cfg: ObsConfig, method: str = "mc",
                 alpha_power: float = 0.7, rc_clip: int = 10):
        assert method in ("mc", "q")
        if obs_cfg.shoe_info == "seen_counts":
            raise ValueError("tabular 不支援 seen_counts，請用 none 或 hilo")
        self.rules, self.obs_cfg, self.method = rules, obs_cfg, method
        self.alpha_power = alpha_power   # 步長 = 1 / N(s,a)^p；p=1 為純平均，p<1 會逐漸淡忘早期資料
        self.rc_clip = rc_clip
        self.n_actions = rules.n_actions
        self.q, self.n = {}, {}

    def _entry(self, k):
        if k not in self.q:
            self.q[k] = np.zeros(self.n_actions)
            self.n[k] = np.zeros(self.n_actions)
        return self.q[k], self.n[k]

    def greedy(self, key, mask) -> int:
        q = self.q.get(key)
        if q is None:
            return int(np.flatnonzero(mask)[0])
        return int(np.where(mask, q, -np.inf).argmax())

    # ---- 與 DQN 相同的介面
    def act(self, env, obs=None) -> int:
        return self.greedy(env_key(env, self.rc_clip), env.legal_mask())

    def first_decision_action(self, cards, dealer_up, mask) -> int:
        k = table_key(self.obs_cfg, "play", cards, False, dealer_up, mask, 0, self.rc_clip)
        return self.greedy(k, mask)

    # ---- 學習
    def learn(self, traj):
        """traj: [(key, action, reward, next_key, next_mask, done), ...]，一局一條。"""
        if self.method == "mc":
            G = sum(t[2] for t in traj)          # gamma = 1，且只有終局獎勵
            for k, a, *_ in traj:
                q, n = self._entry(k)
                n[a] += 1
                q[a] += (G - q[a]) / n[a] ** self.alpha_power
        else:
            for k, a, r, k2, m2, done in traj:
                q, n = self._entry(k)
                n[a] += 1
                target = r
                if not done:
                    q2 = self.q.get(k2)
                    target += 0.0 if q2 is None else np.where(m2, q2, -np.inf).max()
                q[a] += (target - q[a]) / n[a] ** self.alpha_power

    # ---- 存取
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"rules": asdict(self.rules), "obs_cfg": asdict(self.obs_cfg),
                         "method": self.method, "alpha_power": self.alpha_power,
                         "rc_clip": self.rc_clip, "q": self.q, "n": self.n}, f)

    @classmethod
    def load(cls, path) -> "TabularAgent":
        with open(path, "rb") as f:
            d = pickle.load(f)
        rules = RuleConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in d["rules"].items()})
        ag = cls(rules, ObsConfig(**d["obs_cfg"]), d["method"], d["alpha_power"], d["rc_clip"])
        ag.q, ag.n = d["q"], d["n"]
        return ag


def train_tabular(agent: TabularAgent, rounds: int, eps_start=1.0, eps_end=0.01,
                  eps_decay_frac=0.8, seed=0, eval_every=0, eval_rounds=100_000,
                  out_dir=None, log=print, expected_settlement=False):
    from .evaluate import evaluate_policy, format_result
    env = BlackjackEnv(agent.rules, agent.obs_cfg, seed=seed,
                       expected_settlement=expected_settlement)
    rng = np.random.default_rng(seed + 1)
    decay = max(1, int(rounds * eps_decay_frac))
    t0 = time.time()
    for ep in range(1, rounds + 1):
        eps = eps_start + min(1.0, ep / decay) * (eps_end - eps_start)
        env.reset()          # 關閉下注時，無需決策的局已自動結算，不影響學習
        key, traj, done = env_key(env, agent.rc_clip), [], False
        while not done:
            mask = env.legal_mask()
            if rng.random() < eps:
                a = int(rng.choice(np.flatnonzero(mask)))
            else:
                a = agent.greedy(key, mask)
            _, r, done, info = env.step(a)
            if done:
                r = info.get("expected_reward", r)
                k2 = m2 = None
            else:
                k2, m2 = env_key(env, agent.rc_clip), env.legal_mask()
            traj.append((key, a, r, k2, m2, done))
            key = k2
        agent.learn(traj)
        if eval_every and (ep % eval_every == 0 or ep == rounds):
            res = evaluate_policy(agent, agent.rules, agent.obs_cfg, eval_rounds, seed=seed + 999)
            log(f"\nround {ep:,}  eps {eps:.3f}  狀態數 {len(agent.q):,}  ({time.time() - t0:.0f}s)")
            log(format_result(res))
            if out_dir:
                agent.save(os.path.join(out_dir, "tabular.pkl"))
    if out_dir:
        agent.save(os.path.join(out_dir, "tabular.pkl"))
    return agent


def main():
    from .analysis import bet_by_true_count, strategy_table
    from .cli import add_rule_args, obs_from_args, rules_from_args

    p = argparse.ArgumentParser(description="訓練 tabular 對照組")
    add_rule_args(p)
    p.set_defaults(shoe_info="none")
    g = p.add_argument_group("tabular 參數")
    g.add_argument("--method", default="mc", choices=["mc", "q"])
    g.add_argument("--rounds", type=int, default=5_000_000, help="訓練局數")
    g.add_argument("--alpha-power", type=float, default=0.7, help="步長 1/N^p")
    g.add_argument("--eps-end", type=float, default=0.01)
    g.add_argument("--eps-decay-frac", type=float, default=0.8)
    g.add_argument("--rc-clip", type=int, default=10)
    g.add_argument("--eval-every", type=int, default=1_000_000)
    g.add_argument("--eval-rounds", type=int, default=100_000)
    g.add_argument("--expected", action="store_true", help="訓練目標使用期望結算")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", default="runs/tabular")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    agent = TabularAgent(rules_from_args(a), obs_from_args(a), a.method, a.alpha_power, a.rc_clip)
    train_tabular(agent, a.rounds, eps_end=a.eps_end, eps_decay_frac=a.eps_decay_frac,
                  seed=a.seed, eval_every=a.eval_every, eval_rounds=a.eval_rounds, out_dir=a.out,
                  expected_settlement=a.expected)
    strategy_table(agent)
    bet_by_true_count(agent)


if __name__ == "__main__":
    main()
