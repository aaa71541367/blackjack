import numpy as np
import pytest

from blackjack_rl.config import (DOUBLE, HIT, N_PLAY_ACTIONS, SPLIT, STICK,
                                 SURRENDER, ObsConfig, RuleConfig)
from blackjack_rl.env import BlackjackEnv, hand_value
from blackjack_rl.shoe import CARDS_PER_DECK


def make_env(cards, **rule_kw):
    """建立環境並把接下來要發的牌固定成 cards。
    美式發牌順序：玩家1、莊家明牌、玩家2、莊家暗牌、之後依序。"""
    rule_kw.setdefault("enable_betting", True)
    env = BlackjackEnv(RuleConfig(**rule_kw), ObsConfig(), seed=0)
    env.reset()
    env.shoe.stack(cards)
    return env


def bet1(env):
    return env.step(N_PLAY_ACTIONS)  # 下 1 倍注


# ---------------------------------------------------------------- 點數計算（原程式 bug）
@pytest.mark.parametrize("cards,val,soft", [
    ([1, 1], 12, True), ([1, 6, 5], 12, False), ([1, 6], 17, True),
    ([1, 6, 10], 17, False), ([1, 1, 9], 21, True), ([10, 10, 5], 25, False),
])
def test_hand_value(cards, val, soft):
    assert hand_value(cards) == (val, soft)


# ---------------------------------------------------------------- 平手 = 0（原程式算成輸）
def test_push_is_zero():
    env = make_env([10, 10, 8, 8])          # 玩家 18，莊家 18
    bet1(env)
    _, r, done, _ = env.step(STICK)
    assert done and r == 0.0


def test_win_and_lose():
    env = make_env([10, 10, 9, 8]); bet1(env)
    assert env.step(STICK)[1] == 1.0         # 19 vs 18
    env = make_env([10, 10, 7, 8]); bet1(env)
    assert env.step(STICK)[1] == -1.0        # 17 vs 18


# ---------------------------------------------------------------- Double（原程式是 no-op）
def test_double_draws_one_and_ends():
    env = make_env([6, 10, 5, 7, 10])       # 玩家 11，莊家 17，double 抽到 10 → 21
    bet1(env)
    assert env.legal_mask()[DOUBLE]
    _, r, done, _ = env.step(DOUBLE)
    assert done and r == 2.0
    assert len(env.hands[0].cards) == 3


def test_double_bust_loses_double():
    env = make_env([10, 10, 2, 7, 10]); bet1(env)   # 12 double 抽 10 → 22
    _, r, done, _ = env.step(DOUBLE)
    assert done and r == -2.0


def test_double_rule_9_11():
    env = make_env([10, 10, 2, 7], double_rule="9-11"); bet1(env)
    assert not env.legal_mask()[DOUBLE]


# ---------------------------------------------------------------- 分牌（原程式只丟一張牌）
def test_split_two_hands():
    # 玩家 8,8；莊家 10 + 暗牌 7；第一手補 3 → 11，double 抽 10 → 21；第二手補 10 → 18
    env = make_env([8, 10, 8, 7, 3, 10, 10]); bet1(env)
    assert env.legal_mask()[SPLIT]
    _, r, done, _ = env.step(SPLIT)
    assert not done and r == 0.0 and len(env.hands) == 2
    assert env.hands[0].cards == [8, 3] and env.hands[1].cards == [8]   # 第二手尚未補牌
    _, r, done, _ = env.step(DOUBLE)                      # DAS
    assert not done and env.cur == 1 and env.hands[1].cards == [8, 10]
    _, r, done, _ = env.step(STICK)                       # 18 vs 17
    assert done and r == 2.0 + 1.0


def test_split_aces_one_card_only():
    env = make_env([1, 10, 1, 7, 5, 9]); bet1(env)
    _, r, done, _ = env.step(SPLIT)
    # 兩手都只補一張就自動結束：A5=16 輸、A9=20 贏
    assert done and r == 0.0
    assert env.hands[0].cards == [1, 5] and env.hands[1].cards == [1, 9]


def test_split_21_is_not_blackjack():
    env = make_env([1, 7, 1, 10, 10, 10]); bet1(env)     # 莊家 7+10=17
    _, r, done, _ = env.step(SPLIT)
    assert done and r == 2.0                              # 兩手 21 各賠 1:1，不是 1.5


# ---------------------------------------------------------------- 投降
def test_surrender_only_first_decision():
    env = make_env([10, 10, 6, 7, 2]); bet1(env)
    assert env.legal_mask()[SURRENDER]
    env.step(HIT)
    assert not env.legal_mask()[SURRENDER]


def test_surrender_half():
    env = make_env([10, 10, 6, 7]); bet1(env)
    _, r, done, _ = env.step(SURRENDER)
    assert done and r == -0.5


# ---------------------------------------------------------------- Blackjack 與 peek
def test_player_blackjack_pays_3_2():
    env = make_env([1, 9, 10, 7]); _, r, done, _ = bet1(env)
    assert done and r == 1.5


def test_dealer_peek_blackjack_ends_round_without_decision():
    env = make_env([10, 1, 9, 10]); _, r, done, _ = bet1(env)
    assert done and r == -1.0


def test_both_blackjack_push():
    env = make_env([1, 1, 10, 10]); _, r, done, _ = bet1(env)
    assert done and r == 0.0


def test_enhc_dealer_bj_takes_doubled_bet():
    # 歐式：玩家 6,5 + 莊家 A；玩家 double 抽 9；莊家第二張 10 → BJ，輸 2 倍
    env = make_env([6, 1, 5, 9, 10], dealer_peek=False); bet1(env)
    _, r, done, _ = env.step(DOUBLE)
    assert done and r == -2.0


# ---------------------------------------------------------------- 莊家規則
def test_s17_vs_h17():
    for h17, expect in [(False, 0.0), (True, -1.0)]:
        # 玩家 17；莊家 A,6 軟 17；H17 會再補 4 → 21
        env = make_env([10, 1, 7, 6, 4], dealer_hits_soft17=h17); bet1(env)
        assert env.step(STICK)[1] == expect


def test_dealer_does_not_draw_when_player_busts():
    env = make_env([10, 10, 6, 5, 10, 9]); bet1(env)    # 莊家 15，玩家抽 10 爆牌
    pos_before = env.shoe.pos
    _, r, done, _ = env.step(HIT)
    assert done and r == -1.0
    assert env.shoe.pos == pos_before + 1               # 只多發了玩家那張
    assert env.dealer_cards == [10, 5]                  # 暗牌有翻開，但沒有補牌


# ---------------------------------------------------------------- 資訊不外洩
def test_hole_card_not_visible_until_revealed():
    env = make_env([10, 10, 6, 5]); bet1(env)
    seen = env.shoe.seen_counts.copy()
    # 可見：玩家 10、6，莊家明牌 10；暗牌 5 尚未計入
    assert seen[10 - 1] == 2 and seen[6 - 1] == 1 and seen[5 - 1] == 0
    env.shoe.stack([2])
    env.step(HIT)                   # 16 + 2 = 18
    env.step(STICK)                 # 莊家 15 補牌後翻開暗牌
    assert env.shoe.seen_counts[5 - 1] == 1


def test_burn_card_never_visible():
    env = BlackjackEnv(RuleConfig(burn_cards=3), ObsConfig(), seed=1)
    assert env.shoe.pos == 3 and env.shoe.seen_counts.sum() == 0


def test_obs_independent_of_deck_count():
    """同樣看到的牌，4 副和 8 副牌的 state 必須完全相同（state 不含副數與剩餘張數）。"""
    for mode in ("none", "seen_counts", "hilo"):
        obs = []
        for n in (4, 8):
            env = BlackjackEnv(RuleConfig(n_decks=n), ObsConfig(mode), seed=0)
            env.reset()
            env.shoe.stack([10, 9, 6, 7])
            o, *_ = env.step(N_PLAY_ACTIONS)
            obs.append(o)
        assert np.array_equal(obs[0], obs[1]), mode


# ---------------------------------------------------------------- 牌堆
@pytest.mark.parametrize("n", [1, 2, 6, 8])
def test_deck_count_configurable(n):
    env = BlackjackEnv(RuleConfig(n_decks=n), ObsConfig(), seed=0)
    assert env.shoe.n_decks == n
    assert len(env.shoe.cards) == 52 * n
    assert np.array_equal(np.bincount(env.shoe.cards, minlength=11)[1:], CARDS_PER_DECK * n)


def test_random_deck_range():
    env = BlackjackEnv(RuleConfig(n_decks=(4, 8)), ObsConfig(), seed=0)
    seen = set()
    for _ in range(50):
        env.shoe.shuffle()
        seen.add(env.shoe.n_decks)
    assert seen == {4, 5, 6, 7, 8}


def test_reshuffle_only_between_rounds():
    env = BlackjackEnv(RuleConfig(n_decks=1, enable_betting=False), ObsConfig(), seed=3)
    rng = np.random.default_rng(0)
    start = env.shoe.n_shuffles
    for _ in range(3000):
        env.reset()
        shuf, emerg = env.shoe.n_shuffles, env.shoe.n_emergency_shuffles
        done = False
        while not done:
            _, _, done, _ = env.step(rng.choice(np.flatnonzero(env.legal_mask())))
        # 一局進行中只允許緊急重洗，正常重洗只會發生在 reset()
        assert env.shoe.n_shuffles - shuf == env.shoe.n_emergency_shuffles - emerg
        assert env.shoe.seen_counts.sum() <= env.shoe.pos
    assert env.shoe.n_shuffles > start


def test_illegal_action_rejected():
    env = make_env([10, 10, 6, 7]); bet1(env)
    with pytest.raises(AssertionError):
        env.step(SPLIT)             # 不是 pair
    with pytest.raises(AssertionError):
        env.step(99)
