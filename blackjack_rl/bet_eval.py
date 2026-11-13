"""下注模型的驗證：預測能力（校準）與實際下注效果（同一條錄製序列）。

一、預測能力：用「獨立測試牌局」（與訓練不同的 seed）收集資料，目標為每局每單位下注的報酬
   （期望結算，無偏且雜訊較小）。報告：
   - 校準表：依預測值分成 n 個等機率等級，比較各級的預測平均與實際平均（± 標準誤）
   - 校準斜率與截距：實際 = 截距 + 斜率 × 預測；斜率 1、截距 0 為完美校準，
     斜率 < 1 代表預測過度分散（過度自信），> 1 代表預測太保守
   - 相關係數、最高等級與最低等級的實際報酬差

二、實際下注效果：下注不影響發牌，所以在同一條錄製序列（實際輸贏）上，
   所有下注策略打的是完全相同的牌局，差異只來自下注決策。
   所有模型使用相同的「預測值 → 下注」規則（預設：預測為正下最大注，否則最小注），差別只在預測值。
   同一副牌內的局彼此相關，信賴區間用區塊 bootstrap 計算。

  python -m blackjack_rl.bet_eval --decks 4-8 --play runs/full_s1/last.pt \
      --linear runs/full_s1/bet_linear.npz --net runs/full_s1/bet_net.pt --head runs/full_s1/last.pt \
      --calib-rounds 1000000 --calib-seed 3033 --seq runs/bank_test.npz --json results/bet_eval.json
"""
import argparse
import json
import time

import numpy as np

from .bet_model import decide_bet
from .config import N_PLAY_ACTIONS, ObsConfig, RuleConfig
from .env import encode_obs
from .shoe import HILO

_HILO_W = np.array([HILO[c] for c in range(1, 11)])


# ---------------------------------------------------------------- 預測器（輸入：已出現張數陣列）
def make_predictors(rules: RuleConfig, linear=None, net=None, head=None) -> dict:
    """回傳 {名稱: f(seen[n,10]) -> 預測[n]}。head 為兩頭 DQN（value 或 regress 下注頭）。"""
    preds = {}
    if linear is not None:
        from .bet_model import features
        preds["線性下注"] = lambda S: np.array([features(s) for s in S]) @ linear.coef
    if net is not None:
        import torch
        from .bet_learn import _net_input

        def f_net(S):
            with torch.no_grad():
                return net.net(torch.as_tensor(_net_input(S))).squeeze(1).numpy().astype(np.float64)
        preds["神經網路下注"] = f_net
    if head is not None:
        def f_head(S, bs=65536):
            X = np.stack([encode_obs(head.rules, head.obs_cfg, phase="bet", seen_counts=s,
                                     running_count=int(s @ _HILO_W)) for s in S])
            m = np.zeros((1, head.n_actions), bool)
            m[0, N_PLAY_ACTIONS:] = True
            out = []
            for i in range(0, len(X), bs):
                q = head.q_values(X[i:i + bs], np.repeat(m, len(X[i:i + bs]), 0))
                out.append(q[:, N_PLAY_ACTIONS] / head.rules.bet_sizes[0])
            return np.concatenate(out)
        preds["DQN 下注頭"] = f_head
    return preds


# ---------------------------------------------------------------- 一、校準
def calibration(pred, y, n_bins: int = 10) -> dict:
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    edges = np.quantile(pred, np.linspace(0, 1, n_bins + 1))
    k = np.clip(np.searchsorted(edges[1:-1], pred, side="right"), 0, n_bins - 1)
    bins = []
    for i in range(n_bins):
        s = k == i
        if s.sum() < 2:
            continue
        bins.append({"pred": float(pred[s].mean()), "actual": float(y[s].mean()),
                     "se": float(y[s].std(ddof=1) / np.sqrt(s.sum())), "n": int(s.sum())})
    X = np.stack([np.ones_like(pred), pred], 1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    cov = np.linalg.inv(X.T @ X) * resid.var(ddof=2)
    return {"bins": bins, "intercept": float(beta[0]), "slope": float(beta[1]),
            "slope_se": float(np.sqrt(cov[1, 1])), "corr": float(np.corrcoef(pred, y)[0, 1]),
            "top_minus_bottom": bins[-1]["actual"] - bins[0]["actual"],
            "top_minus_bottom_se": float(np.hypot(bins[-1]["se"], bins[0]["se"])),
            "mean_pred": float(pred.mean()), "mean_actual": float(y.mean())}


# ---------------------------------------------------------------- 二、同一條序列上的下注效果
def _block_boot(num, den, block: int, n_boot: int, rng):
    """比值 Σnum/Σden 的區塊 bootstrap 信賴區間（den=None 時為平均）。"""
    n = len(num)
    nb = n // block
    num_b = num[:nb * block].reshape(nb, block).sum(1)
    den_b = (den[:nb * block].reshape(nb, block).sum(1) if den is not None else np.full(nb, block))
    idx = rng.integers(0, nb, (n_boot, nb))
    stats = num_b[idx].sum(1) / den_b[idx].sum(1)
    return [float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))]


def betting_on_sequence(seq, preds: dict, rules: RuleConfig, rule: str = "ev", thresholds=None,
                        block: int = 1000, n_boot: int = 1000, seed: int = 0) -> dict:
    S, R = seq["SEEN"], seq["R"].astype(np.float64)
    sizes = np.array(rules.bet_sizes, float)
    n_levels = len(sizes)
    th = thresholds if thresholds is not None else [0.005 * k for k in range(n_levels - 1)]
    rng = np.random.default_rng(seed)
    bets = {"最小注": np.full(len(R), sizes[0])}
    for name, f in preds.items():
        v = f(S)
        bets[name] = sizes[np.array([decide_bet(e, rules, rule, th) for e in v])]
    out = {}
    for name, b in bets.items():
        g = b * R
        out[name] = {"mean_bet": float(b.mean()), "ev_per_unit": float(g.sum() / b.sum()),
                     "ev_per_unit_ci95": _block_boot(g, b, block, n_boot, rng),
                     "per_round": float(g.mean()), "per_round_ci95": _block_boot(g, None, block, n_boot, rng),
                     "max_bet_rate": float(np.mean(b == sizes[-1]))}
    ref = "線性下注" if "線性下注" in bets else "最小注"
    for name, b in bets.items():                                   # 與參照的配對差（每局）
        d = (b - bets[ref]) * R
        out[name]["diff_per_round_vs_" + ref] = float(d.mean())
        out[name]["diff_ci95"] = _block_boot(d, None, block, n_boot, rng)
    out["_ref"] = ref
    return out


def format_report(cal: dict, bet: dict) -> str:
    L = ["一、預測能力（獨立測試牌局，期望結算）"]
    for name, c in cal.items():
        L.append(f"\n[{name}]  平均預測 {c['mean_pred'] * 100:+.3f}%  實際 {c['mean_actual'] * 100:+.3f}%  "
                 f"相關 {c['corr']:+.4f}")
        L.append(f"  校準：實際 = {c['intercept'] * 100:+.3f}% + {c['slope']:.2f}（±{1.96 * c['slope_se']:.2f}）× 預測"
                 f"    最高等級 − 最低等級 = {c['top_minus_bottom'] * 100:+.2f}%（±{1.96 * c['top_minus_bottom_se'] * 100:.2f}%）")
        L.append("  等級   預測平均   實際平均（±1.96 SE）")
        for i, b in enumerate(c["bins"]):
            L.append(f"  {i + 1:>3}   {b['pred'] * 100:+7.2f}%   {b['actual'] * 100:+7.2f}%（±{1.96 * b['se'] * 100:.2f}%）")
    ref = bet["_ref"]
    L.append(f"\n二、實際下注效果（同一條錄製序列，實際輸贏；區塊 bootstrap 95% CI）")
    L.append(f"{'策略':<14}{'平均下注':>8}{'下最大注':>9}   每單位下注 EV              每局報酬                  相對{ref}（每局）")
    for name, r in bet.items():
        if name.startswith("_"):
            continue
        L.append(f"{name:<14}{r['mean_bet']:>8.2f}{r['max_bet_rate']:>9.1%}   "
                 f"{r['ev_per_unit'] * 100:+.3f}% [{r['ev_per_unit_ci95'][0] * 100:+.2f}, {r['ev_per_unit_ci95'][1] * 100:+.2f}]   "
                 f"{r['per_round'] * 100:+.3f}% [{r['per_round_ci95'][0] * 100:+.2f}, {r['per_round_ci95'][1] * 100:+.2f}]   "
                 f"{r['diff_per_round_vs_' + ref] * 100:+.3f}% [{r['diff_ci95'][0] * 100:+.3f}, {r['diff_ci95'][1] * 100:+.3f}]")
    return "\n".join(L)


def evaluate_bet_models(rules, play, preds, calib_rounds, calib_seed, seq, log=print):
    from .bet_model import collect_bet_data
    t0 = time.time()
    seen, y = collect_bet_data(rules, play, calib_rounds, calib_seed, expected=True, log=log)
    cal = {name: calibration(f(seen), y) for name, f in preds.items()}
    bet = betting_on_sequence(seq, preds, rules)
    if log:
        log(f"（{(time.time() - t0) / 60:.1f} 分鐘）")
    return cal, bet


def main():
    from .agent import DQNAgent
    from .bankroll import load_seq, record_sequence
    from .bet_learn import NeuralBetModel, load_play_policy
    from .bet_model import LinearBetModel
    from .cli import add_rule_args, obs_from_args, rules_from_args
    p = argparse.ArgumentParser(description="下注模型的校準與同序列下注效果")
    add_rule_args(p)
    p.add_argument("--play", default="basic", help="出牌策略：basic、dp 或 DQN 模型")
    p.add_argument("--linear")
    p.add_argument("--net")
    p.add_argument("--head", help="兩頭 DQN 模型（使用其下注頭）")
    p.add_argument("--calib-rounds", type=int, default=1_000_000)
    p.add_argument("--calib-seed", type=int, default=3033, help="獨立測試牌局的 seed（需與訓練不同）")
    p.add_argument("--seq", help="錄製序列（bankroll record 產生）；沒有時自動錄製")
    p.add_argument("--seq-rounds", type=int, default=1_000_000)
    p.add_argument("--seq-seed", type=int, default=4044)
    p.add_argument("--json")
    a = p.parse_args()
    rules, obs_cfg = rules_from_args(a), obs_from_args(a)
    rules = RuleConfig(**{**rules.__dict__, "enable_betting": True})
    play = load_play_policy(a.play, rules)
    if hasattr(play, "obs_cfg"):
        obs_cfg = play.obs_cfg
    head = DQNAgent.load(a.head) if a.head else None
    preds = make_predictors(rules, LinearBetModel.load(a.linear, rules) if a.linear else None,
                            NeuralBetModel.load(a.net, rules) if a.net else None, head)
    if a.seq:
        seq = load_seq(a.seq)
    else:
        print(f"錄製 {a.seq_rounds:,} 局序列（seed {a.seq_seed}）")
        seq = record_sequence(rules, obs_cfg, play, lambda env, obs: 0.0, a.seq_rounds, a.seq_seed)
    cal, bet = evaluate_bet_models(rules, play, preds, a.calib_rounds, a.calib_seed, seq)
    print(format_report(cal, bet))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"calibration": cal, "betting": bet}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
