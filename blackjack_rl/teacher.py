"""老師示範資料：用 basic strategy 或 DP 出牌打一批局，記錄每個出牌決策。

每個決策記錄：
  obs    : state（與 DQN 看到的相同）
  mask   : 合法出牌動作（5 維）
  act    : 老師選的動作
  ev     : 各動作的期望值（DP 老師才有，不合法或未知為 NaN）
並另外記錄可放入 replay buffer 的轉移（每單位下注的報酬，終局用期望結算）。

DP 老師只使用可見資訊（見 optimal.py），所以蒸餾後的 DQN 仍遵守同樣的資訊限制。
DP 的期望值只用在訓練目標，DQN 的輸入仍只有 state。

收集很耗時（4~8 副時每個決策約 2 毫秒），可以用 cache 路徑存檔重複使用。
"""
import os
import time

import numpy as np

from .config import N_PLAY_ACTIONS, ObsConfig, RuleConfig
from .env import BlackjackEnv


def collect_teacher_data(rules: RuleConfig, obs_cfg: ObsConfig, teacher: str, n_rounds: int,
                         seed: int = 777, expected: bool = True, log=print) -> dict:
    from .optimal import optimal_action_evs, visible_info
    from .strategies import BasicStrategy
    assert teacher in ("basic", "dp")
    env = BlackjackEnv(rules, obs_cfg, seed=seed, expected_settlement=expected)
    bs = BasicStrategy()
    O, M, A, EV, TR = [], [], [], [], []
    t0 = time.time()
    for k in range(n_rounds):
        obs, _ = env.reset()
        traj, done = [], False
        while not done:
            if env.phase == "bet":
                obs, r, done, info = env.step(N_PLAY_ACTIONS)      # 只學出牌：最小注
                continue
            m = env.legal_mask()
            if teacher == "dp":
                cards, up, mask, seen, dealt = visible_info(env)
                ev = optimal_action_evs(rules, cards, up, mask, seen, dealt)[:N_PLAY_ACTIONS]
                ev = np.where(m[:N_PLAY_ACTIONS] & np.isfinite(ev), ev, np.nan)
                a = int(np.nanargmax(ev))
            else:
                ev = np.full(N_PLAY_ACTIONS, np.nan)
                a = bs.act(env)
            O.append(obs.copy()); M.append(m[:N_PLAY_ACTIONS].copy()); A.append(a); EV.append(ev)
            o2, r, done, info = env.step(a)
            r_train = (info.get("expected_reward", r) if done else r) / env.base_bet
            traj.append((obs.copy(), a, r_train, o2, float(done), env.legal_mask()))
            obs = o2
        g = sum(t[2] for t in traj)
        TR.extend((*t, g) for t in traj)
        if log and (k + 1) % 50_000 == 0:
            log(f"  老師（{teacher}）已打 {k + 1:,} 局（{time.time() - t0:.0f}s）")
    return {
        "obs": np.array(O, np.float32), "mask": np.array(M, bool), "act": np.array(A, np.int64),
        "ev": np.array(EV, np.float32),
        "tr_obs": np.array([t[0] for t in TR], np.float32), "tr_act": np.array([t[1] for t in TR], np.int64),
        "tr_rew": np.array([t[2] for t in TR], np.float32),
        "tr_obs2": np.array([t[3] for t in TR], np.float32),
        "tr_done": np.array([t[4] for t in TR], np.float32),
        "tr_mask2": np.array([t[5] for t in TR], bool), "tr_g": np.array([t[6] for t in TR], np.float32),
    }


def load_or_collect(path, rules, obs_cfg, teacher, n_rounds, seed=777, expected=True, log=print):
    """有快取就讀取（並檢查老師與局數一致），否則收集後存檔。"""
    from dataclasses import asdict
    sig = repr(asdict(rules)) + repr(asdict(obs_cfg))
    if path and os.path.exists(path):
        d = dict(np.load(path))
        if (str(d.get("teacher")) == teacher and int(d.get("n_rounds", -1)) == n_rounds
                and str(d.get("sig")) == sig):
            log(f"讀取老師示範資料快取：{path}（{len(d['act']):,} 個決策）")
            return d
        log(f"快取 {path} 的老師、局數或規則不同，重新收集")
    t0 = time.time()
    d = collect_teacher_data(rules, obs_cfg, teacher, n_rounds, seed, expected, log)
    log(f"老師示範資料：{n_rounds:,} 局、{len(d['act']):,} 個決策（{time.time() - t0:.0f}s）")
    if path:
        np.savez_compressed(path, teacher=teacher, n_rounds=n_rounds, sig=sig, **d)
    return d
