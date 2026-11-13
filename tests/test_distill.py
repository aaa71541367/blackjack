import numpy as np

from blackjack_rl.config import ObsConfig, RuleConfig, TrainConfig
from blackjack_rl.env import BlackjackEnv
from blackjack_rl.optimal import OptimalPlay
from blackjack_rl.paired import paired_eval
from blackjack_rl.teacher import collect_teacher_data, load_or_collect
from blackjack_rl.train import train

RULES = RuleConfig(n_decks=6, enable_betting=False)
OBS = ObsConfig("seen_counts")


def test_dp_teacher_data_consistent():
    d = collect_teacher_data(RULES, OBS, "dp", 300, seed=1, log=None)
    n = len(d["act"])
    assert n > 300 and d["obs"].shape[0] == n and d["ev"].shape == (n, 5)
    assert np.all(d["act"] == np.nanargmax(d["ev"], axis=1))       # 老師選期望值最大的動作
    assert np.all(np.isnan(d["ev"][~d["mask"]]))                     # 不合法動作沒有期望值
    assert np.all(np.isfinite(d["ev"][d["mask"]]))
    assert len(d["tr_act"]) == n                                     # 每個決策都有一筆轉移


def test_teacher_cache_checks_rules(tmp_path):
    p = str(tmp_path / "t.npz")
    logs = []
    load_or_collect(p, RULES, OBS, "dp", 100, log=logs.append)
    load_or_collect(p, RULES, OBS, "dp", 100, log=logs.append)
    assert any("讀取" in m for m in logs)
    logs.clear()
    load_or_collect(p, RuleConfig(n_decks=4, enable_betting=False), OBS, "dp", 100, log=logs.append)
    assert any("重新收集" in m for m in logs)


def test_distillation_matches_dp(tmp_path):
    cfg = TrainConfig(hidden=(128,), total_steps=0, pretrain_rounds=4000, pretrain_updates=3000,
                      teacher="dp", batch_size=256, lr=1e-3, out_dir=str(tmp_path), eval_every=10**9)
    ag = train(RULES, OBS, cfg)
    env, dp = BlackjackEnv(RULES, OBS, seed=9), OptimalPlay(RULES)
    same = tot = 0
    for _ in range(1500):
        obs, _ = env.reset(); done = False
        while not done:
            a, b = ag.act(env, obs), dp.act(env)
            same += a == b; tot += 1
            obs, _, done, _ = env.step(b)
    assert same / tot > 0.9
    assert (tmp_path / "last.pt").exists()


def test_paired_against_dp_baseline():
    dp = OptimalPlay(RuleConfig(n_decks=6))
    r = paired_eval(dp, RuleConfig(n_decks=6), OBS, 500, seed=2, baseline=dp)
    assert r["mean_diff"] == 0 and r["differ_rate"] == 0


def test_combine_distilled_play_with_bet_head(tmp_path):
    """蒸餾模型（無下注）的出牌頭載入兩頭模型並凍結：出牌完全不變，下注頭有學到資料。"""
    import torch
    from blackjack_rl.agent import DQNAgent
    d1 = tmp_path / "distill"
    train(RULES, OBS, TrainConfig(hidden=(32,), total_steps=0, pretrain_rounds=500,
                                  pretrain_updates=200, teacher="dp", out_dir=str(d1)))
    src = DQNAgent.load(str(d1 / "last.pt"))
    d2 = tmp_path / "full"
    full = train(RuleConfig(n_decks=6), OBS,
                 TrainConfig(hidden=(32,), total_steps=4000, n_envs=16, eval_every=10**9,
                             init_play=str(d1 / "last.pt"), freeze_play=True, bet_batch=64,
                             out_dir=str(d2)))
    for k, v in src.q.state_dict().items():
        assert torch.equal(v, full.q.state_dict()[k])            # 出牌頭完全沒被更新
    assert full.bet_head.n_obs > 500
    env = BlackjackEnv(RuleConfig(n_decks=6), OBS, seed=4)
    for _ in range(200):
        obs, _ = env.reset(); obs, _, done, _ = env.step(5)
        while not done:
            assert full.act(env, obs) == src.act(env, obs)        # 出牌決策相同
            obs, _, done, _ = env.step(full.act(env, obs))
