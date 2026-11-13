"""basic strategy、DP、隨機策略在多個獨立 seed 上的統一評估。

報告兩種數值：
  1. 絕對期望值：每個策略以實際輸贏打 n 局（固定最小注），每單位下注的平均報酬
  2. DP 相對 basic 的配對差：同一牌序上兩者各打一次（期望結算），只記錄報酬差，雜訊小很多
每個 seed 各自報告，再合併（等局數平均，標準誤由各 seed 合成）。

  python -m blackjack_rl.baselines --decks 4-8 --seeds 2027,2028,2029 \
      --rounds 1000000 --dp-rounds 300000 --random-rounds 100000 --json results/baselines.json
"""
import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def _abs_job(job):
    import torch
    torch.set_num_threads(1)
    from .config import ObsConfig, RuleConfig
    from .evaluate import evaluate_policy
    from .optimal import OptimalPlay
    from .strategies import BasicStrategy, RandomPolicy
    rules = RuleConfig(**{**job["rules"], "enable_betting": False})
    obs = ObsConfig(job["shoe_info"])
    pol = {"basic": BasicStrategy(), "dp": OptimalPlay(rules),
           "random": RandomPolicy(job["seed"] + 1)}[job["policy"]]
    t0 = time.time()
    r = evaluate_policy(pol, rules, obs, job["rounds"], job["seed"])
    return {"kind": "absolute", "policy": job["policy"], "seed": job["seed"], "rounds": job["rounds"],
            "ev": float(r["ev_per_unit_bet"]), "ci95": [float(x) for x in r["ci95"]],
            "minutes": (time.time() - t0) / 60}


def _paired_job(job):
    import torch
    torch.set_num_threads(1)
    from .config import ObsConfig, RuleConfig
    from .optimal import OptimalPlay
    from .paired import paired_eval
    rules = RuleConfig(**{**job["rules"], "enable_betting": False})
    t0 = time.time()
    r = paired_eval(OptimalPlay(rules), rules, ObsConfig(job["shoe_info"]), job["rounds"], job["seed"])
    return {"kind": "paired", "policy": "dp_vs_basic", "seed": job["seed"], "rounds": job["rounds"],
            "ev": float(r["mean_diff"]), "ci95": [float(x) for x in r["ci95"]],
            "differ_rate": float(r["differ_rate"]), "minutes": (time.time() - t0) / 60}


def _dispatch(job):
    return _paired_job(job) if job["kind"] == "paired" else _abs_job(job)


def pool_results(rs):
    """合併等局數的多個 seed：平均的平均，標準誤由各次標準誤合成。"""
    m = float(np.mean([r["ev"] for r in rs]))
    se = float(np.sqrt(sum(((r["ci95"][1] - r["ci95"][0]) / 3.92) ** 2 for r in rs)) / len(rs))
    spread = float(np.std([r["ev"] for r in rs], ddof=1)) if len(rs) > 1 else 0.0
    return {"ev": m, "ci95": [m - 1.96 * se, m + 1.96 * se], "n_seeds": len(rs),
            "seed_sd": spread, "mean_se": float(np.mean([(r["ci95"][1] - r["ci95"][0]) / 3.92 for r in rs]))}


def run_baselines(rule_kw: dict, shoe_info: str, seeds, rounds: int, dp_rounds: int,
                  random_rounds: int, paired_rounds: int = None, workers: int = 3) -> dict:
    base = {"rules": rule_kw, "shoe_info": shoe_info}
    jobs = []
    for s in seeds:
        jobs.append({**base, "kind": "absolute", "policy": "basic", "seed": s, "rounds": rounds})
        jobs.append({**base, "kind": "absolute", "policy": "dp", "seed": s, "rounds": dp_rounds})
        jobs.append({**base, "kind": "absolute", "policy": "random", "seed": s, "rounds": random_rounds})
        jobs.append({**base, "kind": "paired", "seed": s, "rounds": paired_rounds or dp_rounds})
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        res = list(ex.map(_dispatch, jobs))
    out = {"per_seed": res, "pooled": {}}
    for key in ("basic", "dp", "random", "dp_vs_basic"):
        rs = [r for r in res if r["policy"] == key]
        out["pooled"][key] = pool_results(rs)
    return out


NAMES = {"basic": "basic strategy（絕對）", "dp": "DP 出牌（絕對）", "random": "隨機策略（絕對）",
         "dp_vs_basic": "DP − basic（配對）"}


def format_baselines(out) -> str:
    lines = [f"{'策略':<24}{'seed':>6}{'局數':>11}   每單位下注期望值（95% CI）"]
    for r in out["per_seed"]:
        lines.append(f"{NAMES[r['policy']]:<24}{r['seed']:>6}{r['rounds']:>11,}   "
                     f"{r['ev'] * 100:+.3f}% [{r['ci95'][0] * 100:+.3f}%, {r['ci95'][1] * 100:+.3f}%]")
    lines.append("")
    lines.append(f"{'合併':<24}{'seed 數':>6}{'':>11}   期望值（95% CI）        seed 間標準差 / 單次標準誤")
    for k, p in out["pooled"].items():
        lines.append(f"{NAMES[k]:<24}{p['n_seeds']:>6}{'':>11}   {p['ev'] * 100:+.3f}% "
                     f"[{p['ci95'][0] * 100:+.3f}%, {p['ci95'][1] * 100:+.3f}%]   "
                     f"{p['seed_sd'] * 100:.3f}% / {p['mean_se'] * 100:.3f}%")
    lines.append("seed 間標準差與單次標準誤相近，代表各 seed 的差異只是抽樣雜訊；明顯較大則表示結果依賴特定牌序。")
    return "\n".join(lines)


def main():
    from .cli import add_rule_args
    p = argparse.ArgumentParser(description="basic、DP、隨機策略的多 seed 統一評估")
    add_rule_args(p)
    p.add_argument("--seeds", default="2027,2028,2029")
    p.add_argument("--rounds", type=int, default=1_000_000, help="basic strategy 的局數")
    p.add_argument("--dp-rounds", type=int, default=300_000, help="DP 的局數（DP 較慢）")
    p.add_argument("--random-rounds", type=int, default=100_000)
    p.add_argument("--paired-rounds", type=int, default=None, help="DP 相對 basic 配對評估的局數")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--json")
    a = p.parse_args()
    rule_kw = {"n_decks": a.decks, "penetration": a.penetration, "burn_cards": a.burn,
               "dealer_peek": not a.enhc, "dealer_hits_soft17": a.h17,
               "blackjack_payout": a.bj_payout, "double_rule": a.double_rule,
               "double_after_split": not a.no_das, "max_hands": a.max_hands,
               "resplit_aces": a.rsa, "hit_split_aces": a.hsa, "late_surrender": not a.no_surrender}
    out = run_baselines(rule_kw, a.shoe_info, [int(x) for x in a.seeds.split(",")], a.rounds,
                        a.dp_rounds, a.random_rounds, a.paired_rounds, a.workers)
    print(format_baselines(out))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
