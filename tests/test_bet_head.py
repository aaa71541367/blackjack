import argparse

import numpy as np
import pytest
import torch

from blackjack_rl.agent import DQNAgent
from blackjack_rl.config import N_PLAY_ACTIONS, ObsConfig, RuleConfig, TrainConfig
from blackjack_rl.env import BlackjackEnv
from blackjack_rl.train import train


def _agent(bet_head="value"):
    rules, obs = RuleConfig(n_decks=6), ObsConfig("seen_counts")
    env = BlackjackEnv(rules, obs, seed=0)
    ag = DQNAgent(env.obs_dim, env.n_actions, rules, obs,
                  TrainConfig(hidden=(16,), bet_head=bet_head))
    return env, ag


def test_bet_q_is_bet_size_times_value():
    env, ag = _agent()
    obs, _ = env.reset()
    q = ag.q_values(obs[None], env.legal_mask()[None])[0]
    v = q[N_PLAY_ACTIONS] / env.rules.bet_sizes[0]
    np.testing.assert_allclose(q[N_PLAY_ACTIONS:], v * np.array(env.rules.bet_sizes), rtol=1e-5)


@pytest.mark.parametrize("bias,expect", [(0.05, -1), (-0.05, 0)])
def test_bet_decision_follows_sign_of_value(bias, expect):
    env, ag = _agent()
    with torch.no_grad():
        ag.q.bet_v.weight.zero_(); ag.q.bet_v.bias.fill_(bias)
    obs, _ = env.reset()
    idx = ag.act(env, obs) - N_PLAY_ACTIONS
    assert idx == (len(env.rules.bet_sizes) - 1 if expect == -1 else 0)


def test_legacy_q_mode_has_no_value_head():
    _, ag = _agent("q")
    assert ag.q.n_bet == 0 and np.all(ag.bet_mult == 1)


@pytest.mark.parametrize("mode", ["value", "regress"])
def test_train_two_heads_and_reload(tmp_path, mode):
    """小型訓練可跑完，且存檔後下注頭設定都保留、輸出一致。"""
    rules, obs = RuleConfig(n_decks=6), ObsConfig("seen_counts")
    cfg = TrainConfig(total_steps=3200, n_envs=16, hidden=(16,), batch_size=64,
                      learning_starts=300, eval_every=3200, eval_rounds=200,
                      out_dir=str(tmp_path), bet_head=mode, bet_batch=64)
    ag = train(rules, obs, cfg)
    ag2 = DQNAgent.load(str(tmp_path / "last.pt"))
    assert ag2.bet_mode == mode
    if mode == "value":
        assert ag2.q.n_bet == len(rules.bet_sizes)
    else:
        assert ag.bet_head.n_obs > 100 and ag2.q_out == N_PLAY_ACTIONS
    x = np.zeros((1, ag.obs_dim), np.float32); m = np.ones((1, ag.n_actions), bool)
    np.testing.assert_allclose(ag.q_values(x, m), ag2.q_values(x, m), rtol=1e-5)


def test_compare_matrix_adds_dqn_bet_column(tmp_path):
    from blackjack_rl.compare import run_matrix
    env, ag = _agent()
    p = tmp_path / "m.pt"; ag.save(str(p))
    a = argparse.Namespace(dp=False, dp_exact=False, play_model=str(p), bet_linear=None,
                           bet_net=None, rounds=100, seed=0)
    res = run_matrix(a, RuleConfig(n_decks=6), ag.obs_cfg)
    assert ("DQN 出牌", "DQN 下注頭") in res and ("basic strategy", "DQN 下注頭") in res


# ---------------------------------------------------------------- regress 下注頭
def _regress_agent(shoe="seen_counts"):
    rules, obs = RuleConfig(n_decks=6), ObsConfig(shoe)
    env = BlackjackEnv(rules, obs, seed=0)
    return env, DQNAgent(env.obs_dim, env.n_actions, rules, obs, TrainConfig(hidden=(16,)))


def test_regress_features_match_linear_bet_model():
    """下注頭的特徵必須和線性下注模型完全相同（同樣的歸納偏差）。"""
    from blackjack_rl.bet_model import features
    env, ag = _regress_agent()
    for _ in range(30):
        obs, _ = env.reset()
        f = ag.bet_head.features(torch.as_tensor(obs[None])).numpy()[0]
        np.testing.assert_allclose(f, features(env.shoe.seen_counts)[1:], atol=1e-5)
        env.step(N_PLAY_ACTIONS)
        while env.phase == "play":
            env.step(0)


def test_regress_rls_recovers_linear_function():
    env, ag = _regress_agent()
    rng = np.random.default_rng(0)
    X, Y = [], []
    w_true = rng.normal(0, 0.05, ag.bet_head.nf + 1)
    for _ in range(3000):
        obs, _ = env.reset()
        f = ag.bet_head.features(torch.as_tensor(obs[None])).numpy()[0]
        X.append(obs); Y.append(w_true[0] + f @ w_true[1:])
        env.step(N_PLAY_ACTIONS)
        while env.phase == "play":
            env.step(0)
    ag.push_bet(np.array(X), np.array(Y))
    ag.bet_head.refit()
    np.testing.assert_allclose(ag.bet_head.w.numpy(), w_true, atol=5e-3)


def test_regress_bet_q_is_size_times_value_and_play_head_has_5_outputs():
    env, ag = _regress_agent()
    obs, _ = env.reset()
    with torch.no_grad():
        ag.bet_head.w[0] = 0.02
    q = ag.q_values(obs[None], env.legal_mask()[None])[0]
    np.testing.assert_allclose(q[N_PLAY_ACTIONS:], 0.02 * np.array(env.rules.bet_sizes), rtol=1e-5)
    assert ag.q(torch.as_tensor(obs[None])).shape[1] == N_PLAY_ACTIONS
    assert ag.act(env, obs) == N_PLAY_ACTIONS + len(env.rules.bet_sizes) - 1


def test_regress_works_with_none_shoe_info():
    env, ag = _regress_agent("none")
    assert ag.bet_head.nf == 0 and ag.bet_head.mlp is None
    obs, _ = env.reset()
    ag.act(env, obs)
