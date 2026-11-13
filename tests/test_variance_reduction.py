import copy

import numpy as np
import pytest

from blackjack_rl.agent import DQNAgent, QNetwork
from blackjack_rl.config import (DOUBLE, N_PLAY_ACTIONS, SPLIT, STICK, ObsConfig,
                                 RuleConfig, TrainConfig)
from blackjack_rl.dealer import dealer_distribution
from blackjack_rl.env import BlackjackEnv, encode_obs, play_feature_dim


@pytest.mark.parametrize("cards", [[6], [10, 6], [1, 6], [1]])
@pytest.mark.parametrize("h17", [False, True])
def test_dealer_distribution_sums_to_one(cards, h17):
    counts = np.array([4] * 9 + [16]) * 2
    d = dealer_distribution(cards, counts, h17)
    assert sum(d.values()) == pytest.approx(1.0)
    assert set(d) <= {17, 18, 19, 20, 21, "bust", "bj"}


def _mc_check(rules, cards, actions, n=4000, keep=0):
    """期望結算 ≈ 把剩餘牌堆重新洗牌後、實際結算的平均。
    keep：最後一個動作會先發給玩家的張數（例如 double 為 1），這些牌不洗。"""
    env = BlackjackEnv(rules, ObsConfig(), seed=5, expected_settlement=True)
    env.reset(); env.shoe.stack(cards); env.step(N_PLAY_ACTIONS)
    for a in actions[:-1]:
        env.step(a)
    base = copy.deepcopy(env)
    _, r, done, info = env.step(actions[-1])
    assert done
    expected = info["expected_reward"]
    rng = np.random.default_rng(0)
    rs = []
    for _ in range(n):
        e = copy.deepcopy(base)
        rng.shuffle(e.shoe.cards[e.shoe.pos + keep:])
        rs.append(e.step(actions[-1])[1])
    rs = np.array(rs)
    se = rs.std() / np.sqrt(n)
    assert abs(rs.mean() - expected) < 4 * se + 1e-9, (rs.mean(), expected, se)
    return expected


def test_expected_settlement_matches_simulation_peek():
    _mc_check(RuleConfig(n_decks=1), [10, 6, 8, 10], [STICK])        # 18 vs 莊家 16


def test_expected_settlement_split_hands():
    # 8,8 分牌後兩手都停牌，期望值要同時考慮兩手
    _mc_check(RuleConfig(n_decks=1), [8, 5, 8, 10, 10, 9], [SPLIT, STICK, STICK])


def test_expected_settlement_enhc_double():
    _mc_check(RuleConfig(n_decks=2, dealer_peek=False), [6, 10, 5, 9], [DOUBLE], keep=1)


def test_expected_settlement_reduces_variance():
    """同樣的局面，期望值目標的變異應遠小於實際結果。"""
    rules = RuleConfig(n_decks=6, enable_betting=False)
    rng = np.random.default_rng(1)
    actual, expect = [], []
    env = BlackjackEnv(rules, ObsConfig(), seed=2, expected_settlement=True)
    for _ in range(3000):
        env.reset()
        done = False
        while not done:
            _, r, done, info = env.step(STICK)
        actual.append(r); expect.append(info["expected_reward"])
    assert np.var(expect) < 0.6 * np.var(actual)
    assert abs(np.mean(expect) - np.mean(actual)) < 0.08


def test_evaluation_env_has_no_expected_reward():
    env = BlackjackEnv(RuleConfig(enable_betting=False), ObsConfig(), seed=0)
    env.reset()
    done = False
    while not done:
        _, _, done, info = env.step(STICK)
    assert "expected_reward" not in info


# ---------------------------------------------------------------- rollout
def _agent(rules, obs=ObsConfig("none")):
    env = BlackjackEnv(rules, obs, seed=0, expected_settlement=True)
    cfg = TrainConfig(hidden=(16,), target_mode="rollout", bet_head="q")
    return env, DQNAgent(env.obs_dim, env.n_actions, rules, obs, cfg)


def test_rollout_bet_targets_scale_with_bet():
    """所有分支共用同樣的牌序，而出牌策略不看下注，所以下注 b 倍的目標剛好是 b 倍。"""
    env, ag = _agent(RuleConfig(bet_sizes=(1, 2, 4, 8)))
    for _ in range(20):
        env.reset()
        tgt, mask = ag.rollout_targets(env)
        assert mask[N_PLAY_ACTIONS:].all() and not mask[:N_PLAY_ACTIONS].any()
        np.testing.assert_allclose(tgt[N_PLAY_ACTIONS:], tgt[N_PLAY_ACTIONS] * np.array([1, 2, 4, 8]),
                                   rtol=1e-5, atol=1e-6)


def test_rollout_does_not_change_env():
    env, ag = _agent(RuleConfig())
    env.reset(); env.step(N_PLAY_ACTIONS)
    pos, obs = env.shoe.pos, env.observe().copy()
    if env.phase == "play":
        tgt, mask = ag.rollout_targets(env)
        assert env.shoe.pos == pos and np.array_equal(env.observe(), obs)
        assert np.all(tgt[~mask] == 0)


# ---------------------------------------------------------------- one-hot / dueling
def test_onehot_encoding():
    rules = RuleConfig()
    m = np.ones(rules.n_actions, bool)
    for enc, dim in [("onehot", 37), ("scalar", 20)]:
        cfg = ObsConfig("none", enc)
        assert play_feature_dim(cfg) == dim
        x = encode_obs(rules, cfg, phase="play", hand_cards=[10, 3], dealer_up=6, mask=m)
        assert len(x) == dim
    x = encode_obs(rules, ObsConfig("none"), phase="play", hand_cards=[10, 3], dealer_up=6, mask=m)
    assert x[1 + 13 - 4] == 1.0 and x[1:19].sum() == 1.0


def test_dueling_shapes_and_identifiability():
    import torch
    net = QNetwork(10, 9, (16,), dueling=True)
    q = net(torch.zeros(4, 10))
    assert q.shape == (4, 9)
