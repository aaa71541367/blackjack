"""用相同 seed 與局數比較多個策略。

一、比較矩陣（出牌 × 下注），分開回答兩個問題：
      「DQN 出牌 vs DP 出牌」：看同一欄（相同下注方式）的不同列
      「學到的下注 vs 線性下注」：看同一列（相同出牌策略）的不同欄
  python -m blackjack_rl.compare --decks 6 --dp --play-model runs/play/best.pt \\
      --bet-linear runs/play/bet_linear.npz --bet-net runs/play/bet_net.pt --rounds 1000000

  列（出牌）：basic strategy、DP（--dp，選用）、DQN 的出牌頭（--play-model）
  欄（下注）：最小注、線性下注（--bet-linear）、神經網路下注（--bet-net）、
              DQN 的下注頭（--play-model 為兩頭模型時自動加入）

二、直接比較完整模型（舊用法，.pt 為 DQN、.pkl 為 tabular）：
  python -m blackjack_rl.compare runs/td/best.pt runs/tab/tabular.pkl --rounds 500000
"""
import argparse

from .analysis import load_any, strategy_table
from .evaluate import evaluate_policy
from .strategies import BasicStrategy


def _fmt(res):
    lo, hi = res["ci95"]
    return res["ev_per_unit_bet"] * 100, lo * 100, hi * 100, res["mean_bet"]


def run_matrix(a, rules, obs_cfg):
    from .bet_learn import NeuralBetModel
    from .bet_model import BaselinePolicy, LinearBetModel
    from .optimal import OptimalPlay

    plays = [("basic strategy", BasicStrategy())]
    if a.dp:
        plays.append(("DP 出牌", OptimalPlay(rules, exact=a.dp_exact)))
    if a.play_model:
        from .agent import DQNAgent
        ag = DQNAgent.load(a.play_model)
        plays.append(("DQN 出牌", ag))

    bets = [("最小注", None)]
    if rules.enable_betting:
        if a.bet_linear:
            bets.append(("線性下注", LinearBetModel.load(a.bet_linear, rules)))
        if a.bet_net:
            bets.append(("學到的下注", NeuralBetModel.load(a.bet_net, rules)))
        if a.play_model and ag.bet_mode in ("value", "regress"):
            from .bet_model import DQNBetHead
            bets.append(("DQN 下注頭", DQNBetHead(ag)))

    print(f"每格：每單位下注 EV（每局報酬 95% CI）平均下注；{a.rounds:,} 局，seed {a.seed}\n")
    results = {}
    for pname, play in plays:
        agree = strategy_table(play, verbose=False) if pname != "basic strategy" else 1.0
        for bname, bm in bets:
            res = evaluate_policy(BaselinePolicy(play, bm), rules, obs_cfg, a.rounds, a.seed)
            ev, lo, hi, mb = _fmt(res)
            results[(pname, bname)] = ev
            print(f"{pname:<16}{bname:<10}{ev:+8.3f}%   [{lo:+.3f}%, {hi:+.3f}%]   "
                  f"平均下注 {mb:.2f}   出牌一致率 {agree:.1%}", flush=True)

    print("\n摘要")
    if ("DQN 出牌", "最小注") in results and ("DP 出牌", "最小注") in results:
        d = results[("DQN 出牌", "最小注")] - results[("DP 出牌", "最小注")]
        print(f"  DQN 出牌 − DP 出牌（最小注）：{d:+.3f}%")
    for pname, _ in plays:
        for learned in ("學到的下注", "DQN 下注頭"):
            if (pname, learned) in results and (pname, "線性下注") in results:
                d = results[(pname, learned)] - results[(pname, "線性下注")]
                print(f"  {learned} − 線性下注（{pname}）：{d:+.3f}%")
    print("  注意：差異需大於信賴區間寬度才有意義；同一個 seed 可降低但無法消除雜訊。")
    return results


def main():
    from .cli import add_rule_args, obs_from_args, rules_from_args
    p = argparse.ArgumentParser(description="比較多個策略")
    add_rule_args(p)
    p.add_argument("models", nargs="*", help="完整模型：.pt（DQN）或 .pkl（tabular）")
    p.add_argument("--rounds", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--dp", action="store_true", help="加入 DP 出牌")
    p.add_argument("--dp-exact", action="store_true", help="DP 使用精確模式（較慢）")
    p.add_argument("--play-model", help="只學出牌的 DQN 模型（矩陣模式）")
    p.add_argument("--bet-linear", help="線性下注模型 .npz（矩陣模式）")
    p.add_argument("--bet-net", help="神經網路下注模型 .pt（矩陣模式）")
    p.add_argument("--bet-model", help="舊參數名稱，等同 --bet-linear")
    a = p.parse_args()
    a.bet_linear = a.bet_linear or a.bet_model

    if a.play_model or a.bet_linear or a.bet_net or (a.dp and not a.models):
        rules = rules_from_args(a)
        if a.play_model:
            from .agent import DQNAgent
            obs_cfg = DQNAgent.load(a.play_model).obs_cfg   # 環境需產生 DQN 看得懂的 state
        else:
            obs_cfg = obs_from_args(a)
        run_matrix(a, rules, obs_cfg)
        return

    # ---- 舊用法：直接比較完整模型
    agents = [(m, load_any(m)) for m in a.models]
    rules = agents[0][1].rules if agents else rules_from_args(a)
    for m, ag in agents[1:]:
        if ag.rules != rules:
            print(f"警告：{m} 的規則與第一個模型不同，結果不可直接比較")
    obs0 = agents[0][1].obs_cfg if agents else obs_from_args(a)
    rows = [("basic strategy", BasicStrategy(), obs0, None)]
    for m, ag in agents:
        rows.append((m, ag, ag.obs_cfg, ag))
    print(f"{'策略':<40}{'每單位下注 EV':>14}{'95% CI（每局）':>26}{'一致率':>9}{'平均下注':>9}")
    for label, pol, obs_cfg, ag in rows:
        res = evaluate_policy(pol, rules, obs_cfg, a.rounds, a.seed)
        ev, lo, hi, mb = _fmt(res)
        agree = f"{strategy_table(ag, verbose=False):.1%}" if ag is not None else "100%"
        print(f"{label:<40}{ev:>+13.3f}%   [{lo:+.3f}%, {hi:+.3f}%]{agree:>9}{mb:>9.2f}", flush=True)


if __name__ == "__main__":
    main()
