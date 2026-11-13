"""配對評估：在完全相同的牌序上比較出牌策略與 basic strategy。

出牌能改善的空間只有約 0.1%~0.3%，一般評估的雜訊（每局 ±1 以上）會完全蓋過它。
這裡每局開始時複製環境，讓候選策略與 basic strategy 在「同一副牌序」上各打一次，
只記錄兩者報酬的差。兩者決策相同的局差值恰為 0，只有決策不同的局才貢獻雜訊，
因此能以少很多的局數偵測出很小的差距。

  - 只比較出牌：兩者都下最小注
  - 報酬預設用期望結算（對莊家補牌取條件期望），這是期望值的無偏估計，只是雜訊更小
  - 主環境由 basic strategy 推進，所以衡量的是「在 basic strategy 會遇到的牌堆狀態下，
    改用候選策略出牌的收益」

用法：
  python -m blackjack_rl.paired runs/a/best.pt runs/b/best.pt --dp --rounds 300000
"""
import argparse
import copy
import time

import numpy as np

from .config import N_PLAY_ACTIONS, ObsConfig, RuleConfig
from .env import BlackjackEnv
from .strategies import BasicStrategy


def _play_round(env, obs, policy):
    acts, done = [], False
    while not done:
        a = int(policy.act(env, obs))
        acts.append(a)
        obs, r, done, info = env.step(a)
    return info.get("expected_reward", r) / env.base_bet, acts


def paired_eval(policy, rules: RuleConfig, obs_cfg: ObsConfig, n_rounds: int = 200_000,
                seed: int = 777, expected: bool = True, baseline=None) -> dict:
    baseline = baseline or BasicStrategy()
    env = BlackjackEnv(rules, obs_cfg, seed=seed, expected_settlement=expected)
    diffs = np.zeros(n_rounds)
    differ = np.zeros(n_rounds, bool)
    k = 0
    while k < n_rounds:
        obs, info = env.reset()
        k += len(info["auto_rewards"])            # 無需決策的局：兩者結果相同，差值為 0
        if k >= n_rounds:
            break
        if env.phase == "bet":
            obs, r, done, _ = env.step(N_PLAY_ACTIONS)   # 只比較出牌：固定最小注
            if done:
                k += 1
                continue
        e_pol = copy.deepcopy(env)
        g_pol, a_pol = _play_round(e_pol, obs, policy)
        g_base, a_base = _play_round(env, obs, baseline)   # 主環境由 basic 推進
        diffs[k] = g_pol - g_base
        differ[k] = a_pol != a_base
        k += 1
    se = diffs.std(ddof=1) / np.sqrt(n_rounds)
    return {
        "mean_diff": diffs.mean(),
        "ci95": (diffs.mean() - 1.96 * se, diffs.mean() + 1.96 * se),
        "differ_rate": differ.mean(),
        "diff_when_differ": diffs[differ].mean() if differ.any() else 0.0,
        "rounds": n_rounds,
    }


def main():
    from .analysis import load_any
    from .cli import add_rule_args, obs_from_args, rules_from_args
    from .optimal import OptimalPlay

    p = argparse.ArgumentParser(description="與 basic strategy 做配對的出牌比較")
    add_rule_args(p)
    p.add_argument("models", nargs="*", help="DQN 模型（.pt）")
    p.add_argument("--dp", action="store_true", help="加入 DP 出牌（只用可見資訊的近似上限）")
    p.add_argument("--rounds", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--actual", action="store_true", help="用實際結算而非期望結算（雜訊較大）")
    p.add_argument("--baseline", default="basic", choices=["basic", "dp"],
                   help="比較基準：basic strategy 或 DP 出牌（檢驗是否超越老師）")
    p.add_argument("--thresholds", default=None,
                   help="對每個 DQN 模型套用多個決策門檻比較，例如 0,0.02,0.05（不需重新訓練）")
    a = p.parse_args()

    agents = [(m, load_any(m)) for m in a.models]
    rules = agents[0][1].rules if agents else rules_from_args(a)
    obs_cfg = agents[0][1].obs_cfg if agents else obs_from_args(a)
    cands = []
    for m, ag in agents:
        if a.thresholds:
            for t in (float(x) for x in a.thresholds.split(",")):
                ag2 = copy.copy(ag)
                ag2.cfg = copy.copy(ag.cfg)
                ag2.cfg.deviate_threshold = t
                ag2._sel_prior = None
                cands.append((f"{m} τ={t:g}", ag2))
        else:
            cands.append((m, ag))
    if a.dp:
        cands.append(("DP 出牌", OptimalPlay(rules)))

    base = OptimalPlay(rules) if a.baseline == "dp" else None
    bname = "DP 出牌" if a.baseline == "dp" else "basic strategy"
    print(f"與 {bname} 的配對差（每局每單位下注）；{a.rounds:,} 局，seed {a.seed}")
    print(f"{'策略':<34}{'平均差':>10}{'95% CI':>26}{'決策不同的局':>12}{'不同時的平均差':>14}")
    for name, pol in cands:
        t0 = time.time()
        res = paired_eval(pol, rules, obs_cfg, a.rounds, a.seed, not a.actual, baseline=base)
        lo, hi = res["ci95"]
        short = "DP" if a.baseline == "dp" else "basic"
        verdict = f"優於 {short}" if lo > 0 else (f"劣於 {short}" if hi < 0 else "無法區分")
        print(f"{name:<34}{res['mean_diff'] * 100:>+9.3f}%   [{lo * 100:+.3f}%, {hi * 100:+.3f}%]"
              f"{res['differ_rate']:>11.1%}{res['diff_when_differ'] * 100:>+13.2f}%   {verdict}"
              f"（{time.time() - t0:.0f}s）", flush=True)


if __name__ == "__main__":
    main()
