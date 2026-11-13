import numpy as np
import pytest

from blackjack_rl.agent import DQNAgent
from blackjack_rl.config import ObsConfig, RuleConfig, TrainConfig
from blackjack_rl.env import BlackjackEnv
from blackjack_rl.paired import paired_eval
from blackjack_rl.strategies import BasicStrategy, RandomPolicy
from blackjack_rl.train import train


def _agent(**kw):
    rules, obs = RuleConfig(n_decks=6, enable_betting=False), ObsConfig("seen_counts")
    env = BlackjackEnv(rules, obs, seed=0)
    return env, DQNAgent(env.obs_dim, env.n_actions, rules, obs, TrainConfig(hidden=(32,), **kw))


def _agreement(env, ag, n=3000):
    bs, same, tot = BasicStrategy(), 0, 0
    for _ in range(n):
        obs, _ = env.reset()
        done = False
        while not done:
            a, b = ag.act(env, obs), bs.act(env)
            same += a == b; tot += 1
            obs, _, done, _ = env.step(b)
    return same / tot


def test_residual_starts_exactly_at_basic_strategy():
    env, ag = _agent(play_prior="residual")
    assert _agreement(env, ag) == 1.0


def test_pretraining_imitates_basic_strategy(tmp_path):
    rules, obs = RuleConfig(n_decks=6, enable_betting=False), ObsConfig("seen_counts")
    cfg = TrainConfig(hidden=(64,), total_steps=64, n_envs=8, pretrain_rounds=6000,
                      pretrain_updates=1500, batch_size=256, learning_starts=10**9,
                      eval_every=10**9, out_dir=str(tmp_path), lr=1e-3)
    ag = train(rules, obs, cfg)
    assert _agreement(BlackjackEnv(rules, obs, seed=3), ag) > 0.9


def test_paired_eval_identity_and_random():
    rules, obs = RuleConfig(n_decks=6), ObsConfig("seen_counts")
    same = paired_eval(BasicStrategy(), rules, obs, 2000, seed=1)
    assert same["mean_diff"] == 0 and same["differ_rate"] == 0
    rnd = paired_eval(RandomPolicy(0), rules, obs, 3000, seed=1)
    assert rnd["ci95"][1] < 0 and rnd["differ_rate"] > 0.5


def test_residual_training_smoke(tmp_path):
    rules, obs = RuleConfig(n_decks=6), ObsConfig("seen_counts")
    cfg = TrainConfig(total_steps=3200, n_envs=16, hidden=(16,), batch_size=64,
                      learning_starts=300, eval_every=3200, eval_rounds=200, bet_batch=64,
                      out_dir=str(tmp_path), play_prior="residual", eps_start=0.1)
    train(rules, obs, cfg)
    ag = DQNAgent.load(str(tmp_path / "last.pt"))
    assert ag.prior is not None


def test_deviate_threshold_only_affects_selection():
    """門檻很大時，任何模型都完全照 basic strategy；門檻不改變網路輸出。"""
    import torch
    env, ag = _agent()
    obs, _ = env.reset()
    raw = ag.q(torch.as_tensor(obs[None])).detach().clone()
    ag.cfg.deviate_threshold = 100.0
    assert _agreement(env, ag, 1000) == 1.0
    assert torch.equal(ag.q(torch.as_tensor(obs[None])), raw)


def test_legal_from_obs_matches_env():
    import torch
    from blackjack_rl.prior import BasicPrior
    rules, obs_cfg = RuleConfig(n_decks=6), ObsConfig("seen_counts")
    env = BlackjackEnv(rules, obs_cfg, seed=5)
    pr, rp = BasicPrior(rules, obs_cfg), RandomPolicy(1)
    n = 0
    for _ in range(3000):
        obs, _ = env.reset(); done = False
        while not done:
            if env.phase == "play":
                m = env.legal_mask()[:5]
                g = pr.legal_from_obs(torch.as_tensor(obs[None]))[0].numpy()
                h = env.hands[env.cur]
                if not (h.from_aces and not rules.hit_split_aces):   # 分 A 鎖定的手無法由 state 判斷
                    assert np.array_equal(m, g); n += 1
            obs, _, done, _ = env.step(rp.act(env))
    assert n > 1000


def test_save_every_and_anchor_keeps_basic(tmp_path):
    """從零開始（無預訓練），只靠強的間距損失，模型也應該被拉向 basic strategy。"""
    rules, obs = RuleConfig(n_decks=6, enable_betting=False), ObsConfig("seen_counts")
    cfg = TrainConfig(hidden=(64,), total_steps=24000, n_envs=16, batch_size=256, lr=1e-3,
                      learning_starts=500, train_every=2, eval_every=10**9, save_every=8000,
                      anchor_weight=5.0, eps_start=0.3, out_dir=str(tmp_path))
    ag = train(rules, obs, cfg)
    assert all((tmp_path / f"step_{k}.pt").exists() for k in (8000, 16000, 24000))
    assert _agreement(BlackjackEnv(rules, obs, seed=3), ag) > 0.9
