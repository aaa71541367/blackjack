"""評估任一策略：平均每局報酬、每單位下注期望值、95% 信賴區間、下注與動作分布。"""
import argparse
import json
from collections import Counter

import numpy as np

from .config import N_PLAY_ACTIONS, ObsConfig, RuleConfig
from .env import BlackjackEnv

ACTION_NAMES = ["stick", "hit", "surrender", "split", "double"]


def evaluate_policy(policy, rules: RuleConfig, obs_cfg: ObsConfig,
                    n_rounds: int = 100_000, seed: int = 12345) -> dict:
    env = BlackjackEnv(rules, obs_cfg, seed=seed)
    rewards, bets = [], []
    actions, bet_choice = Counter(), Counter()

    while len(rewards) < n_rounds:
        obs, info = env.reset()
        for r in info["auto_rewards"]:          # 關閉下注時，無需決策的局也要算
            rewards.append(r)
            bets.append(1.0)
        done = False
        while not done:
            a = int(policy.act(env, obs))
            if a >= N_PLAY_ACTIONS:
                bet_choice[rules.bet_sizes[a - N_PLAY_ACTIONS]] += 1
            else:
                actions[ACTION_NAMES[a]] += 1
            obs, r, done, info = env.step(a)
        rewards.append(r)
        bets.append(info["base_bet"])

    rw = np.asarray(rewards[:n_rounds], dtype=np.float64)
    bt = np.asarray(bets[:n_rounds], dtype=np.float64)
    se = rw.std(ddof=1) / np.sqrt(len(rw))
    n_act = sum(actions.values()) or 1
    return {
        "policy": getattr(policy, "name", type(policy).__name__),
        "rounds": len(rw),
        "mean_reward_per_round": rw.mean(),
        "ci95": (rw.mean() - 1.96 * se, rw.mean() + 1.96 * se),
        "ev_per_unit_bet": rw.sum() / bt.sum(),   # 報酬 ÷ 初始下注總額
        "mean_bet": bt.mean(),
        "bet_distribution": {str(k): v / max(1, sum(bet_choice.values()))
                             for k, v in sorted(bet_choice.items())},
        "action_distribution": {k: v / n_act for k, v in actions.items()},
        "emergency_shuffles": env.shoe.n_emergency_shuffles,
    }


def format_result(res: dict) -> str:
    lo, hi = res["ci95"]
    lines = [
        f"[{res['policy']}] {res['rounds']:,} 局",
        f"  平均每局報酬 : {res['mean_reward_per_round']:+.4f}  (95% CI {lo:+.4f} ~ {hi:+.4f})",
        f"  每單位下注 EV: {res['ev_per_unit_bet'] * 100:+.3f}%   平均下注 {res['mean_bet']:.3f}",
    ]
    if res["bet_distribution"]:
        lines.append("  下注分布     : " + ", ".join(f"{k}x {v:.1%}" for k, v in res["bet_distribution"].items()))
    lines.append("  動作分布     : " + ", ".join(f"{k} {v:.1%}" for k, v in res["action_distribution"].items()))
    return "\n".join(lines)


def main():
    from .cli import add_rule_args, rules_from_args, obs_from_args
    from .strategies import BasicStrategy, RandomPolicy

    p = argparse.ArgumentParser(description="評估策略")
    add_rule_args(p)
    p.add_argument("--policy", default="basic", choices=["basic", "random", "dqn"])
    p.add_argument("--model", help="DQN 模型檔（--policy dqn 時使用）")
    p.add_argument("--rounds", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    rules, obs_cfg = rules_from_args(a), obs_from_args(a)

    if a.policy == "basic":
        pol = BasicStrategy()
    elif a.policy == "random":
        pol = RandomPolicy(a.seed)
    else:
        from .agent import DQNAgent
        pol = DQNAgent.load(a.model)
        rules, obs_cfg = pol.rules, pol.obs_cfg      # 用模型訓練時的設定

    res = evaluate_policy(pol, rules, obs_cfg, a.rounds, a.seed)
    print(json.dumps(res, indent=2, default=str) if a.json else format_result(res))


if __name__ == "__main__":
    main()
