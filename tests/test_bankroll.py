import numpy as np

from blackjack_rl.bankroll import (BankrollConfig, BankrollSim, compare_policies, make_dp_policy,
                                   policy_bold, policy_flat, policy_greedy, run_policy, solve_dp,
                                   train_dqn, make_dqn_policy, load_dqn)


def _seq(n=20000, p_win=0.5, seed=0):
    rng = np.random.default_rng(seed)
    R = np.where(rng.random(n) < p_win, 1.0, -1.0).astype(np.float32)
    V = rng.normal(0, 0.01, n).astype(np.float32)
    return {"V": V, "R": R, "SEEN": np.zeros((n, 10), np.int16)}


def test_sim_bookkeeping():
    cfg = BankrollConfig(start=10, target=20, horizon=5, bets=(1, 2, 4))
    seq = {"V": np.zeros(10, np.float32), "R": np.array([1, 1, -1, 1, 1, 1, 1, 1, 1, 1], np.float32),
           "SEEN": np.zeros((10, 10), np.int16)}
    sim = BankrollSim(seq, cfg, np.array([0]))
    for a, b_exp in [(2, 14), (2, 18), (1, 16), (2, 20)]:
        win, done = sim.step(np.array([a]))
        assert sim.b[0] == b_exp
    assert sim.success[0] and sim.done[0]


def test_ruin_and_deadline():
    cfg = BankrollConfig(start=4, target=100, horizon=3, bets=(1, 4))
    lose = {"V": np.zeros(10, np.float32), "R": -np.ones(10, np.float32), "SEEN": np.zeros((10, 10), np.int16)}
    assert not run_policy(lambda s: np.ones(s.n, np.int64), lose, cfg, np.array([0])).any()   # 一把破產
    fair = _seq(100)
    assert not run_policy(policy_flat, fair, cfg, np.array([0, 10])).any()                   # 期限到未達標


def test_bold_beats_flat_in_fair_game_and_dp_is_best():
    """公平遊戲、期限短時，最小注幾乎不可能達標；DP 應不劣於其他策略。"""
    cfg = BankrollConfig(start=10, target=20, horizon=40, bets=(1, 2, 4, 8))
    seq = _seq(60000, 0.49)
    dp = solve_dp(seq, cfg, n_buckets=4, log=None)
    res = compare_policies({"flat": policy_flat, "期望值為正下最大注（目前規則）": policy_greedy,
                            "bold": policy_bold, "dp": make_dp_policy(dp)}, seq, cfg, 4000, 1)
    assert res["bold"]["success"] > res["flat"]["success"] + 0.1
    assert res["dp"]["success"] >= res["bold"]["success"] - 0.03


def test_train_dqn_smoke(tmp_path):
    cfg = BankrollConfig(start=10, target=20, horizon=40, bets=(1, 2, 4, 8))
    seq = _seq(30000, 0.49)
    model, hist = train_dqn(seq, cfg, episodes=3000, n_envs=64, batch=256, buffer=50000,
                            eval_seq=seq, eval_episodes=500, eval_every=1500, out_dir=str(tmp_path), log=None)
    m2, c2 = load_dqn(str(tmp_path / "model.pt"))
    starts = np.arange(0, 20000, 100)
    assert np.array_equal(run_policy(make_dqn_policy(model), seq, cfg, starts),
                          run_policy(make_dqn_policy(m2), seq, cfg, starts))
    assert len(hist) >= 1


def test_outcomes_drawdown_and_final():
    from blackjack_rl.bankroll import run_episodes
    cfg = BankrollConfig(start=10, target=20, horizon=4, bets=(1, 2, 4))
    R = np.array([1, 1, -1, -1,  -1, -1, -1, -1,  1, -1, 1, -1], np.float32)
    seq = {"V": np.zeros(12, np.float32), "R": R, "SEEN": np.zeros((12, 10), np.int16)}
    # 每局想下 4：起點 0 → 14, 18, 14, 10（到期）
    # 起點 4 → 6, 2，之後資金不足 4 只能下最小注 1 → 1, 0（破產）
    ep = run_episodes(lambda s: np.full(s.n, 2), seq, cfg, np.array([0, 4]))
    assert ep["timeout"][0] and ep["ruin"][1] and not ep["success"].any()
    assert ep["final"][0] == 10 and ep["mdd"][0] == 8 and abs(ep["mdd_frac"][0] - 8 / 18) < 1e-9
    assert ep["mdd"][1] == 10 and ep["final"][1] == 0 and ep["length"].tolist() == [4, 4]


def test_disjoint_starts_do_not_overlap():
    from blackjack_rl.bankroll import disjoint_starts
    cfg = BankrollConfig(horizon=50)
    st = disjoint_starts(_seq(10000), cfg)
    assert np.all(np.diff(st) >= cfg.horizon) and st[-1] + cfg.horizon < 10000


def test_dqn_policy_uses_model_normalization():
    """模型以期限 1000 訓練，用在期限 500 時，狀態應以模型自己的期限正規化。"""
    import torch
    from blackjack_rl.bankroll import BankrollQ, make_dqn_policy, make_obs
    mcfg = BankrollConfig(start=50, target=100, horizon=1000, bets=(1, 2))
    seen = []

    class Spy(BankrollQ):
        def forward(self, x):
            seen.append(x.clone()); return super().forward(x)
    m = Spy(3, 2)
    ecfg = BankrollConfig(start=50, target=100, horizon=500, bets=(1, 2))
    sim = BankrollSim(_seq(2000), ecfg, np.array([0]))
    make_dqn_policy(m, mcfg)(sim)
    assert abs(float(seen[0][0, 1]) - 500 / 1000) < 1e-6                  # 剩 500 局 / 模型期限 1000


def test_compare_policies_report_fields():
    cfg = BankrollConfig(start=10, target=20, horizon=40, bets=(1, 2, 4, 8))
    res = compare_policies({"期望值為正下最大注（目前規則）": policy_greedy, "flat": policy_flat},
                           _seq(20000, 0.49), cfg, starts="random", n_episodes=300)
    r = res["flat"]
    assert abs(r["success"] + r["ruin"] + r["timeout"] - 1) < 1e-9
    assert set(r["final_q"]) == {"5", "25", "50", "75", "95"} and r["ci95"][0] <= r["success"] <= r["ci95"][1]
