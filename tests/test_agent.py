import numpy as np

from blackjack_rl.agent import DQNAgent, ReplayBuffer
from blackjack_rl.config import ObsConfig, RuleConfig, TrainConfig
from blackjack_rl.env import BlackjackEnv


def _agent(rules=None, obs=None):
    rules, obs = rules or RuleConfig(), obs or ObsConfig()
    env = BlackjackEnv(rules, obs, seed=0)
    cfg = TrainConfig(hidden=(32,), batch_size=16, target_update=5)
    return env, DQNAgent(env.obs_dim, env.n_actions, rules, obs, cfg)


def test_masked_actions_always_legal():
    env, agent = _agent()
    rng = np.random.default_rng(0)
    for _ in range(300):
        obs, _ = env.reset()
        done = False
        while not done:
            m = env.legal_mask()
            a = agent.act_batch(obs[None], m[None], eps=0.3, rng=rng)[0]
            assert m[a]
            obs, _, done, _ = env.step(a)


def test_update_and_save_load(tmp_path):
    env, agent = _agent(RuleConfig(n_decks=(4, 8)), ObsConfig("hilo"))
    buf = ReplayBuffer(1000, env.obs_dim, env.n_actions)
    rng = np.random.default_rng(0)
    obs, _ = env.reset()
    for _ in range(200):
        a = rng.choice(np.flatnonzero(env.legal_mask()))
        o2, r, d, _ = env.step(a)
        if a < 5:                     # 與訓練迴圈相同：regress 模式下注轉移不進出牌 buffer
            buf.push(obs, a, r, o2, float(d), env.legal_mask())
        obs = env.reset()[0] if d else o2
    for _ in range(20):
        loss = agent.update(buf.sample(16, rng))
        assert np.isfinite(loss)
    path = tmp_path / "m.pt"
    agent.save(str(path))
    loaded = DQNAgent.load(str(path))
    assert loaded.rules == agent.rules and loaded.obs_cfg == agent.obs_cfg
    m = np.ones((1, env.n_actions), bool)
    assert np.allclose(agent.q_values(obs[None], m), loaded.q_values(obs[None], m))
