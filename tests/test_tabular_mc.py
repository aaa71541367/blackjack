import numpy as np
import pytest

from blackjack_rl.agent import DQNAgent, ReplayBuffer
from blackjack_rl.config import ObsConfig, RuleConfig, TrainConfig
from blackjack_rl.env import BlackjackEnv
from blackjack_rl.strategies import basic_action
from blackjack_rl.tabular import TabularAgent, env_key, train_tabular


def test_tabular_rejects_seen_counts():
    with pytest.raises(ValueError):
        TabularAgent(RuleConfig(), ObsConfig("seen_counts"))


def test_key_has_no_deck_info():
    keys = []
    for n in (4, 8):
        env = BlackjackEnv(RuleConfig(n_decks=n), ObsConfig("hilo"), seed=0)
        env.reset(); env.shoe.stack([10, 9, 6, 7]); env.step(5)
        keys.append(env_key(env))
    assert keys[0] == keys[1]


@pytest.mark.parametrize("method", ["mc", "q"])
def test_tabular_learns_obvious_decisions(method, tmp_path):
    rules = RuleConfig(n_decks=6, enable_betting=False)
    ag = TabularAgent(rules, ObsConfig("none"), method=method)
    train_tabular(ag, 60_000, seed=1)
    m = np.array([True, True, True, False, True])
    assert ag.first_decision_action([10, 10], 6, np.array([True, True, True, True, True])) == 0
    assert ag.first_decision_action([10, 9], 7, m) == 0          # 硬 19 停牌
    assert ag.first_decision_action([6, 5], 6, m) in (1, 4)      # 11 對莊家 6 應抽牌或加倍
    p = tmp_path / "t.pkl"; ag.save(str(p))
    assert TabularAgent.load(str(p)).q.keys() == ag.q.keys()


def test_mc_return_stored_for_every_step():
    """train.py 的做法：一局結束後，每一步都存入整局報酬 G。"""
    rules = RuleConfig(enable_betting=True)
    env = BlackjackEnv(rules, ObsConfig(), seed=3)
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(100, env.obs_dim, env.n_actions)
    obs, _ = env.reset(); pending = []
    done = False
    while not done:
        a = rng.choice(np.flatnonzero(env.legal_mask()))
        o2, r, done, _ = env.step(a)
        pending.append((obs.copy(), a, r, o2, float(done), env.legal_mask())); obs = o2
    G = sum(t[2] for t in pending)
    for t in pending:
        buf.push(*t, g=G)
    assert np.all(buf.ret[:buf.n] == G) and buf.n == len(pending) >= 2


def test_mc_update_and_lr_decay():
    rules, obs = RuleConfig(), ObsConfig()
    env = BlackjackEnv(rules, obs, seed=0)
    cfg = TrainConfig(hidden=(16,), batch_size=8, target_mode="mc", lr=1e-3, lr_end=1e-4,
                      total_steps=100, learning_starts=0, train_every=1, bet_head="q")
    ag = DQNAgent(env.obs_dim, env.n_actions, rules, obs, cfg)
    buf = ReplayBuffer(50, env.obs_dim, env.n_actions)
    for _ in range(50):
        buf.push(np.zeros(env.obs_dim), 5, 0.0, np.zeros(env.obs_dim), 1.0,
                 np.zeros(env.n_actions, bool), g=1.0)
    rng = np.random.default_rng(0)
    for _ in range(100):
        ag.update(buf.sample(8, rng))
    assert ag.opt.param_groups[0]["lr"] == pytest.approx(1e-4, rel=1e-3)
    # MC 目標為常數 1 時，Q 應往 1 / max_bet 靠近（reward 依最大下注縮放）
    q = ag.q_values(np.zeros((1, env.obs_dim), np.float32), np.ones((1, env.n_actions), bool))[0, 5]
    assert abs(q - 1 / 8) < abs(0 - 1 / 8)
