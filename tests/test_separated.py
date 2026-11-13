import argparse

import numpy as np

from blackjack_rl.agent import DQNAgent
from blackjack_rl.bet_learn import NeuralBetModel
from blackjack_rl.bet_model import BaselinePolicy, LinearBetModel, collect_bet_data
from blackjack_rl.config import N_PLAY_ACTIONS, ObsConfig, RuleConfig, TrainConfig
from blackjack_rl.env import BlackjackEnv
from blackjack_rl.strategies import BasicStrategy


def _play_only_agent():
    rules = RuleConfig(n_decks=6, enable_betting=False)
    obs = ObsConfig("seen_counts")
    env = BlackjackEnv(rules, obs, seed=0)
    return DQNAgent(env.obs_dim, env.n_actions, rules, obs, TrainConfig(hidden=(16,)))


def test_play_only_dqn_runs_in_betting_env():
    """只學出牌的 DQN 放進有下注的環境：state 維度相同、下注時下最小注、出牌動作都合法。"""
    ag = _play_only_agent()
    env = BlackjackEnv(RuleConfig(n_decks=6), ObsConfig("seen_counts"), seed=1)
    assert env.obs_dim == ag.obs_dim and env.n_actions > ag.n_actions
    for _ in range(200):
        obs, _ = env.reset()
        assert ag.act(env, obs) == N_PLAY_ACTIONS
        obs, r, done, _ = env.step(N_PLAY_ACTIONS)
        while not done:
            a = ag.act(env, obs)
            assert a < N_PLAY_ACTIONS and env.legal_mask()[a]
            obs, r, done, _ = env.step(a)


def test_collect_and_fit_both_bet_models_on_same_data(tmp_path):
    rules = RuleConfig(n_decks=2, penetration=(0.75, 0.75))
    seen, y = collect_bet_data(rules, BasicStrategy(), 30_000, seed=0, log=None)
    assert seen.shape == (30_000, 10) and y.shape == (30_000,)
    low = np.zeros(10, int); low[1:6] = 8
    high = np.zeros(10, int); high[9] = 30; high[0] = 10
    # 真實資料：線性模型在 3 萬局就能學到方向
    lin = LinearBetModel(rules).fit_arrays(seen, y)
    assert lin.predict(low) > lin.predict(high)
    # 神經網路在真實資料上需要更多局數，這裡用訊號明確的人造資料檢查訓練流程本身
    hilo = np.array([-1, 1, 1, 1, 1, 1, 0, 0, 0, -1])
    y_syn = 0.01 * (seen @ hilo) + np.random.default_rng(0).normal(0, 0.05, len(seen))
    net = NeuralBetModel(rules, hidden=(16,)).fit_arrays(seen, y_syn, epochs=30, log=None)
    assert net.predict(low) > net.predict(np.zeros(10, int)) > net.predict(high)
    p = tmp_path / "n.pt"; net.save(str(p))
    net2 = NeuralBetModel.load(str(p), rules)
    assert abs(net2.predict(low) - net.predict(low)) < 1e-6


def test_baseline_policy_combines_play_and_bet():
    rules = RuleConfig(n_decks=6)
    ag = _play_only_agent()

    class AlwaysMax:
        name = "max"
        def bet_index(self, seen):
            return len(rules.bet_sizes) - 1

    pol = BaselinePolicy(ag, AlwaysMax())
    env = BlackjackEnv(rules, ObsConfig("seen_counts"), seed=2)
    obs, _ = env.reset()
    assert pol.act(env, obs) == N_PLAY_ACTIONS + len(rules.bet_sizes) - 1


def test_compare_matrix_smoke(tmp_path):
    from blackjack_rl.compare import run_matrix
    ag = _play_only_agent()
    mp = tmp_path / "play.pt"; ag.save(str(mp))
    rules = RuleConfig(n_decks=6)
    seen, y = collect_bet_data(rules, BasicStrategy(), 3000, seed=0, log=None)
    LinearBetModel(rules).fit_arrays(seen, y).save(str(tmp_path / "l.npz"))
    NeuralBetModel(rules, hidden=(8,)).fit_arrays(seen, y, epochs=2, log=None).save(str(tmp_path / "n.pt"))
    a = argparse.Namespace(dp=True, dp_exact=False, play_model=str(mp),
                           bet_linear=str(tmp_path / "l.npz"), bet_net=str(tmp_path / "n.pt"),
                           rounds=200, seed=0)
    res = run_matrix(a, rules, ag.obs_cfg)
    assert len(res) == 9          # 3 種出牌 × 3 種下注
