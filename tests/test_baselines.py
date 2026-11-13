import numpy as np
import pytest

from blackjack_rl.analysis import strategy_table
from blackjack_rl.bet_model import BaselinePolicy, LinearBetModel, features
from blackjack_rl.config import N_PLAY_ACTIONS, ObsConfig, RuleConfig
from blackjack_rl.env import BlackjackEnv
from blackjack_rl.optimal import OptimalPlay, deck_posterior, optimal_action_evs, visible_info
from blackjack_rl.strategies import BasicStrategy


def test_dp_reproduces_basic_strategy_fresh_shoe():
    """basic strategy 本身就是由這類計算得出，剛洗好的多副牌上應完全一致。"""
    assert strategy_table(OptimalPlay(RuleConfig(n_decks=6)), verbose=False) == 1.0


def test_deck_posterior():
    rules = RuleConfig(n_decks=(4, 8))
    post = deck_posterior(rules, np.zeros(10, int), 1)
    assert sum(post.values()) == pytest.approx(1)
    # 已看到 170 張 → 4 副（208 張）時已超過 80%，早就該發到 cut card，不可能是 4 副
    seen = np.array([13] * 9 + [53])
    post = deck_posterior(rules, seen, int(seen.sum()) + 1)
    assert 4 not in post or post[4] < 1e-6
    # 已看到 20 張 A → 至少 5 副
    seen = np.zeros(10, int); seen[0] = 20
    assert all(n >= 5 for n, w in deck_posterior(rules, seen, 21).items() if w > 1e-9)


def test_dp_uses_only_visible_info():
    """規則先驗相同（4~8 副）、看到同樣的牌時，真實牌堆是 4 副或 8 副，
    DP 的期望值都必須完全相同。"""
    evs = []
    for n in (4, 8):
        env = BlackjackEnv(RuleConfig(n_decks=(4, 8)), ObsConfig(), seed=0)
        while env.shoe.n_decks != n:
            env.shoe.shuffle()
        env.reset(); env.shoe.stack([10, 9, 6, 7]); env.step(N_PLAY_ACTIONS)
        assert env.shoe.n_decks == n
        c, u, m, s, db = visible_info(env)
        evs.append(optimal_action_evs(env.rules, c, u, m, s, db))
    np.testing.assert_allclose(evs[0], evs[1])


def test_fast_vs_exact_close():
    r = RuleConfig(n_decks=2)
    seen = np.array([2, 3, 3, 4, 4, 2, 1, 2, 1, 8])
    m = np.array([True, True, True, False, True])
    for cards, up in [([10, 6], 10), ([6, 5], 6), ([1, 7], 9)]:
        a = optimal_action_evs(r, cards, up, m, seen, 40, exact=False)
        b = optimal_action_evs(r, cards, up, m, seen, 40, exact=True)
        ok = np.isfinite(a) & np.isfinite(b)          # 非法動作為 -inf，不比較
        assert np.max(np.abs(a[ok] - b[ok])) < 0.02


def test_bet_model_learns_count_direction():
    """小型訓練：看到較多小牌（2~6）時預測期望值應較高，看到較多 10 點與 A 時較低。"""
    rules = RuleConfig(n_decks=2, penetration=(0.75, 0.75))
    bm = LinearBetModel(rules).fit(BasicStrategy(), 40_000, seed=0, log=None)
    low = np.zeros(10, int); low[1:6] = 8          # 看到 40 張小牌
    high = np.zeros(10, int); high[9] = 30; high[0] = 10
    assert bm.predict(low) > bm.predict(np.zeros(10, int)) > bm.predict(high)
    assert bm.bet_index(low) == len(rules.bet_sizes) - 1 and bm.bet_index(high) == 0


def test_baseline_policy_bets_min_without_model():
    env = BlackjackEnv(RuleConfig(), ObsConfig(), seed=0)
    env.reset()
    assert BaselinePolicy(BasicStrategy()).act(env) == N_PLAY_ACTIONS
    from blackjack_rl.bet_model import N_FEATURES
    assert len(features(np.zeros(10))) == N_FEATURES == 11
