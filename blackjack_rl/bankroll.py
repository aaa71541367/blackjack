"""資金管理：有限資金、期限內把資金翻倍的機率（展示 RL 的序列決策）。

問題：起始資金 start 單位，在 horizon 局內達到 target 單位就成功；資金低於最小注即破產。
每局必須下注（不能坐著看），下注額不得超過目前資金。目標是最大化成功機率。

為什麼這是 RL 問題：最佳下注取決於牌面優勢、目前資金與剩餘局數。落後且時間不多時，
即使期望值略負也值得冒險；接近目標時應保守。現在的下注會改變未來的資金，
這種跨時間的取捨是單局期望值迴歸做不到的。

實作重點：下注不影響發牌，所以「每局的牌面優勢與每單位結果」這條序列與下注策略無關，
事先錄製一次（record），之後 DQN 訓練、DP 求解、各策略評估都重複使用，
而且所有策略可以在完全相同的牌序上配對比較。結果使用實際輸贏（不是期望結算），
因為破產風險取決於真實的波動。
簡化：分牌、加倍的額外下注不受資金限制（損失超過資金時視為破產）。

子指令：
  record   錄製序列：每局的優勢預測 V、每單位實際結果 R、已出現張數
  dp       在「資金 × 剩餘局數 × 優勢等級」網格上求最佳解（上限參照）
  train    訓練資金管理 DQN
  evaluate 在測試序列上比較各策略的成功機率
  heatmap  畫出 DQN 與 DP 在不同資金、剩餘局數下的下注
"""
import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import N_PLAY_ACTIONS, ObsConfig, RuleConfig
from .env import BlackjackEnv, _SEEN_SCALE


@dataclass
class BankrollConfig:
    start: float = 50.0
    target: float = 100.0
    horizon: int = 500
    bets: Tuple[int, ...] = (1, 2, 4, 8, 16)
    min_bet: float = 1.0
    edge_scale: float = 20.0          # V（約 ±5%）乘上此值作為 DQN 輸入
    edge_input: str = "v"             # DQN 看到的牌面資訊："v"（預測優勢）、"shoe"（已出現張數）、"both"


# =============================================================================
# 錄製
# =============================================================================
class _EdgeFromAgent:
    """用兩頭 DQN 的下注頭（每單位期望值）作為優勢預測。"""

    def __init__(self, agent):
        self.agent = agent

    def __call__(self, env, obs):
        q = self.agent.q_values(obs[None], env.legal_mask()[None])[0]
        return float(q[N_PLAY_ACTIONS] / env.rules.bet_sizes[0])


class _EdgeFromLinear:
    def __init__(self, model):
        self.model = model

    def __call__(self, env, obs):
        return self.model.predict(env.shoe.seen_counts)


def record_sequence(rules: RuleConfig, obs_cfg: ObsConfig, play, edge_fn, n_rounds: int,
                    seed: int, log=print) -> dict:
    """以固定的出牌策略連續打 n_rounds 局（牌堆跨局延續），記錄每局的優勢預測與實際結果。"""
    rules = RuleConfig(**{**rules.__dict__, "enable_betting": True})
    assert abs(rules.blackjack_payout * 2 - round(rules.blackjack_payout * 2)) < 1e-9, \
        "資金網格以 0.5 單位為單位，Blackjack 賠率需為 0.5 的倍數"
    env = BlackjackEnv(rules, obs_cfg, seed=seed)
    V = np.zeros(n_rounds, np.float32)
    R = np.zeros(n_rounds, np.float32)
    SEEN = np.zeros((n_rounds, 10), np.int16)
    t0 = time.time()
    for k in range(n_rounds):
        obs, _ = env.reset()
        SEEN[k] = env.shoe.seen_counts
        V[k] = edge_fn(env, obs)
        obs, r, done, _ = env.step(N_PLAY_ACTIONS)            # 以 1 單位打這局
        while not done:
            obs, r, done, _ = env.step(play.act(env, obs))
        R[k] = r / rules.bet_sizes[0]
        if log and (k + 1) % 250_000 == 0:
            log(f"  已錄製 {k + 1:,} 局（{time.time() - t0:.0f}s）")
    return {"V": V, "R": R, "SEEN": SEEN}


def load_seq(path):
    d = np.load(path)
    return {k: d[k] for k in ("V", "R", "SEEN")}


# =============================================================================
# 向量化模擬器
# =============================================================================
def make_obs(cfg: BankrollConfig, b, t_left, v, seen):
    parts = [np.stack([b / cfg.target, t_left / cfg.horizon], 1)]
    if cfg.edge_input in ("v", "both"):
        parts.append(np.clip(v * cfg.edge_scale, -3, 3)[:, None])
    if cfg.edge_input in ("shoe", "both"):
        parts.append(seen / _SEEN_SCALE[None, :])
    return np.concatenate(parts, 1).astype(np.float32)


def obs_dim(cfg: BankrollConfig):
    return 2 + (1 if cfg.edge_input in ("v", "both") else 0) + (10 if cfg.edge_input in ("shoe", "both") else 0)


class BankrollSim:
    """同時模擬多個資金 episode，每個 episode 從錄製序列的某個起點連續取局。
    記錄每個 episode 的結局（成功／破產／期限到期）、期末資金、最高點與最大回撤。"""

    RUNNING, SUCCESS, RUIN, TIMEOUT = 0, 1, 2, 3

    def __init__(self, seq, cfg: BankrollConfig, starts: np.ndarray):
        self.seq, self.cfg = seq, cfg
        self.bets = np.array(cfg.bets, np.float64)
        self.n = len(starts)
        self.pos = starts.astype(np.int64).copy()
        self.b = np.full(self.n, float(cfg.start))
        self.t = np.zeros(self.n, np.int64)
        self.done = np.zeros(self.n, bool)
        self.success = np.zeros(self.n, bool)
        self.outcome = np.zeros(self.n, np.int8)
        self.peak = self.b.copy()
        self.mdd = np.zeros(self.n)               # 最大回撤（單位）：max_t (最高點 − 目前資金)
        self.mdd_frac = np.zeros(self.n)          # 最大回撤（比例）：max_t (最高點 − 目前資金) / 最高點

    def edge(self):
        return self.seq["V"][self.pos].astype(np.float64)

    def seen(self):
        return self.seq["SEEN"][self.pos]

    def obs(self):
        return make_obs(self.cfg, self.b, self.cfg.horizon - self.t, self.edge(), self.seen())

    def mask(self):
        return self.bets[None, :] <= self.b[:, None] + 1e-9

    def step(self, a):
        alive = ~self.done
        r = self.seq["R"][self.pos].astype(np.float64)
        self.b = np.where(alive, self.b + self.bets[a] * r, self.b)
        self.t = np.where(alive, self.t + 1, self.t)
        self.pos = np.where(alive, self.pos + 1, self.pos)
        self.peak = np.where(alive, np.maximum(self.peak, self.b), self.peak)
        dd = self.peak - np.maximum(self.b, 0.0)
        self.mdd = np.where(alive, np.maximum(self.mdd, dd), self.mdd)
        self.mdd_frac = np.where(alive, np.maximum(self.mdd_frac, dd / self.peak), self.mdd_frac)
        win = alive & (self.b >= self.cfg.target - 1e-9)
        ruin = alive & ~win & (self.b < self.cfg.min_bet - 1e-9)
        tout = alive & ~win & ~ruin & (self.t >= self.cfg.horizon)
        self.outcome[win], self.outcome[ruin], self.outcome[tout] = self.SUCCESS, self.RUIN, self.TIMEOUT
        self.success |= win
        self.done |= win | ruin | tout
        return win, win | ruin | tout

    def restart(self, i, start_pos):
        self.pos[i], self.b[i], self.t[i] = start_pos, self.cfg.start, 0
        self.done[i] = self.success[i] = False


def random_starts(seq, cfg, n, rng):
    return rng.integers(0, len(seq["R"]) - cfg.horizon - 1, n)


def disjoint_starts(seq, cfg, max_n=None):
    """互不重疊的起點：每個 episode 最多用 horizon 局，起點間隔 horizon，彼此不共用牌局。"""
    st = np.arange(0, len(seq["R"]) - cfg.horizon - 1, cfg.horizon)
    return st[:max_n] if max_n else st


# =============================================================================
# 基準策略（輸入：模擬器，輸出：每個 episode 的下注 index）
# =============================================================================
def _largest_allowed(bets, amount, mask):
    """不超過 amount 的最大合法下注；若沒有則最小注。"""
    ok = (bets[None, :] <= amount[:, None] + 1e-9) & mask
    idx = np.where(ok, np.arange(len(bets))[None, :], -1).max(1)
    return np.maximum(idx, 0)


def policy_flat(sim):
    return np.zeros(sim.n, np.int64)


def policy_greedy(sim):
    """目前系統的規則：優勢為正就下最大（可負擔的）注，否則最小注。"""
    m = sim.mask()
    top = np.where(m, np.arange(m.shape[1])[None, :], -1).max(1)
    return np.where(sim.edge() > 0, top, 0)


def policy_kelly(sim, var=1.3):
    """Kelly 比例下注：下注 ≈ 資金 × 優勢 / 變異數；優勢不為正時最小注。"""
    amt = sim.b * np.maximum(sim.edge(), 0) / var
    return _largest_allowed(sim.bets, amt, sim.mask())


def policy_bold(sim):
    """Bold play：下注剛好能達標的金額（不超過資金），與優勢無關。"""
    amt = np.minimum(sim.b, sim.cfg.target - sim.b)
    return _largest_allowed(sim.bets, amt, sim.mask())


REF = "期望值為正下最大注（目前規則）"
BASELINES = {"固定最小注": policy_flat, REF: policy_greedy,
             "Kelly 比例下注": policy_kelly, "Bold play（下注到達標所需）": policy_bold}


def run_episodes(policy, seq, cfg, starts) -> dict:
    """執行所有 episode，回傳每個 episode 的結局與路徑統計。"""
    sim = BankrollSim(seq, cfg, starts)
    while not sim.done.all():
        a = policy(sim)
        a = np.where(sim.mask()[np.arange(sim.n), a], a, 0)
        sim.step(a)
    return {"success": sim.outcome == sim.SUCCESS, "ruin": sim.outcome == sim.RUIN,
            "timeout": sim.outcome == sim.TIMEOUT, "final": sim.b.copy(), "mdd": sim.mdd.copy(),
            "mdd_frac": sim.mdd_frac.copy(), "length": sim.t.copy()}


def run_policy(policy, seq, cfg, starts):
    return run_episodes(policy, seq, cfg, starts)["success"]


# =============================================================================
# DP 上限參照
# =============================================================================
def solve_dp(seq, cfg: BankrollConfig, n_buckets: int = 12, log=print) -> dict:
    """把優勢 V 分成等機率的等級 k，由錄製序列估計 P(r, k' | k)（本局結果與下一局等級），
    在資金（0.5 單位網格）× 剩餘局數 × 等級上做逆向歸納。這是近似 MDP 的最佳解：
    假設優勢等級是馬可夫的，且本局結果只透過等級與過去相關。"""
    t0 = time.time()
    V, R = seq["V"].astype(np.float64), seq["R"].astype(np.float64)
    edges = np.quantile(V, np.linspace(0, 1, n_buckets + 1)[1:-1])
    k = np.searchsorted(edges, V)
    r2 = np.round(R * 2).astype(np.int64)                     # 結果以 0.5 單位計
    rvals = np.unique(r2)
    ri = np.searchsorted(rvals, r2)
    K, NR = n_buckets, len(rvals)
    P = np.zeros((K, NR, K))
    np.add.at(P, (k[:-1], ri[:-1], k[1:]), 1.0)
    P /= np.maximum(P.sum((1, 2), keepdims=True), 1)
    nb = int(round(cfg.target * 2)) + 1                        # 資金 0, 0.5, ..., target
    bgrid = np.arange(nb) / 2.0
    bets = np.array(cfg.bets)
    alive = bgrid >= cfg.min_bet - 1e-9
    idx = [np.clip(np.arange(nb)[:, None] + bb * rvals[None, :], 0, nb - 1) for bb in bets]
    allowed = bets[None, :] <= bgrid[:, None] + 1e-9           # (nb, A)
    Vt = np.zeros((nb, K))
    Vt[-1, :] = 1.0                                            # 剩 0 局：只有已達標者成功
    pi = np.zeros((cfg.horizon + 1, nb, K), np.int8)
    ar = np.arange(NR)[None, :]
    for tl in range(1, cfg.horizon + 1):
        Vn = np.zeros((nb, K))
        for kk in range(K):
            A = Vt @ P[kk].T                                   # (nb, NR)：A[b', r] = Σ_k' P(r,k'|k) V(b', k')
            Q = np.stack([A[idx[a], ar].sum(1) for a in range(len(bets))], 1)
            Q = np.where(allowed, Q, -np.inf)
            pi[tl, :, kk] = Q.argmax(1)
            Vn[:, kk] = np.where(alive, Q.max(1), 0.0)
        Vn[-1, :] = 1.0
        Vt = Vn
    kstart = np.bincount(k, minlength=K) / len(k)
    p_model = float(Vt[int(round(cfg.start * 2))] @ kstart)
    if log:
        log(f"DP 求解完成（{time.time() - t0:.0f}s）：{K} 個優勢等級、{NR} 種單局結果，"
            f"模型預估成功機率 {p_model:.2%}")
    return {"pi": pi, "edges": edges, "p_model": p_model}


def make_dp_policy(dp):
    pi, edges = dp["pi"], dp["edges"]

    def policy(sim):
        tl = np.clip(sim.cfg.horizon - sim.t, 0, pi.shape[0] - 1)
        bi = np.clip(np.round(sim.b * 2).astype(np.int64), 0, pi.shape[1] - 1)
        kk = np.searchsorted(edges, sim.edge())
        return pi[tl, bi, kk].astype(np.int64)
    return policy


# =============================================================================
# 資金管理 DQN
# =============================================================================
class BankrollQ(nn.Module):
    """dueling=True 時 Q = V(s) + A(s,a) − mean A：狀態價值（成功率 0~1）差異很大，
    但同一狀態下不同下注的差距很小，分開學習較容易抓到細微差距。"""

    def __init__(self, d_in, n_act, hidden=(128, 128), dueling=True):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.dueling = dueling
        if dueling:
            self.v, self.a = nn.Linear(d, 1), nn.Linear(d, n_act)
        else:
            self.out = nn.Linear(d, n_act)

    def forward(self, x):
        h = self.trunk(x)
        if self.dueling:
            a = self.a(h)
            return self.v(h) + a - a.mean(1, keepdim=True)
        return self.out(h)


def make_dqn_policy(model, model_cfg: BankrollConfig = None):
    """model_cfg：模型訓練時的設定。狀態以模型自己的方式正規化（資金 / 模型的目標、剩餘局數 /
    模型的期限），所以同一個模型可以用在不同的起始資金與期限（敏感度分析）。"""
    @torch.no_grad()
    def policy(sim):
        if model_cfg is None:
            o = sim.obs()
        else:
            seen = sim.seen() if hasattr(sim, "seen") else np.zeros((sim.n, 10))
            o = make_obs(model_cfg, sim.b, sim.cfg.horizon - sim.t, sim.edge(), seen)
        q = model(torch.as_tensor(o)).numpy()
        return np.where(sim.mask(), q, -np.inf).argmax(1)
    return policy


def train_dqn(seq, cfg: BankrollConfig, episodes: int = 60_000, n_envs: int = 512,
              hidden=(128, 128), lr: float = 1e-3, lr_end: float = 1e-4, batch: int = 2048,
              buffer: int = 2_000_000, eps_start: float = 1.0, eps_end: float = 0.02,
              target: str = "td", explore_starts: float = 0.5, target_update: int = 1000,
              dueling: bool = True, updates_per_step: int = 4,
              eval_seq=None, eval_episodes: int = 5000, eval_every: int = 10_000,
              seed: int = 0, out_dir: str = None, log=print):
    """Q(s, a) 估計「在 s 下注 a 後最終達標的機率」，回報稀疏（只在結束時給 1 或 0）。

    target：
      "td"：Double DQN（gamma = 1）。學的是貪婪策略的價值，不受探索動作污染（預設）
      "mc"：每個轉移的目標是它所屬 episode 最後是否成功（含探索動作的行為策略價值）
    explore_starts：訓練時有此比例的 episode 從隨機的資金與剩餘局數開始，
      讓「接近期限」等少見狀態也有足夠資料。狀態只由資金、剩餘局數與牌面決定，
      所以從任何狀態出發都是合法的子問題。評估一律從標準起點開始。"""
    assert target in ("td", "mc")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    d_in, n_act = obs_dim(cfg), len(cfg.bets)
    model = BankrollQ(d_in, n_act, hidden, dueling)
    tgt = BankrollQ(d_in, n_act, hidden, dueling)
    tgt.load_state_dict(model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    est_updates = max(1, episodes * 120 // n_envs * updates_per_step)     # 平均 episode 約 120 局
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: 1 + min(1, u / est_updates) * (lr_end / lr - 1))
    O = np.zeros((buffer, d_in), np.float32)
    A = np.zeros(buffer, np.int64)
    G = np.zeros(buffer, np.float32)                           # mc：回報；td：本步獎勵
    O2 = np.zeros((buffer, d_in), np.float32) if target == "td" else None
    D = np.zeros(buffer, np.float32) if target == "td" else None
    M2 = np.zeros((buffer, n_act), bool) if target == "td" else None
    ptr, size = 0, 0

    def new_start(k):
        pos = random_starts(seq, cfg, k, rng)
        b = np.full(k, float(cfg.start))
        t = np.zeros(k, np.int64)
        es = rng.random(k) < explore_starts
        b[es] = np.round(rng.uniform(cfg.min_bet, cfg.target, es.sum()) * 2) / 2
        t[es] = rng.integers(0, cfg.horizon, es.sum())
        return pos, b, t

    sim = BankrollSim(seq, cfg, np.zeros(n_envs, np.int64))
    sim.pos, sim.b, sim.t = new_start(n_envs)
    ep_o = [[] for _ in range(n_envs)]
    ep_a = [[] for _ in range(n_envs)]
    finished, next_eval, n_upd, t0 = 0, eval_every, 0, time.time()
    eval_starts = disjoint_starts(eval_seq, cfg, eval_episodes) if eval_seq is not None else None
    history = []
    while finished < episodes:
        eps = eps_start + min(1.0, finished / max(1, episodes * 0.6)) * (eps_end - eps_start)
        o, m = sim.obs(), sim.mask()
        with torch.no_grad():
            q = model(torch.as_tensor(o)).numpy()
        a = np.where(m, q, -np.inf).argmax(1)
        for i in np.flatnonzero(rng.random(sim.n) < eps):
            a[i] = rng.choice(np.flatnonzero(m[i]))
        if target == "mc":
            for i in range(sim.n):
                ep_o[i].append(o[i]); ep_a[i].append(a[i])
        win, done = sim.step(a)
        if target == "td":                                     # 向量化寫入所有轉移
            j = (ptr + np.arange(sim.n)) % buffer
            O[j], A[j], G[j] = o, a, win.astype(np.float32)
            O2[j], D[j], M2[j] = sim.obs(), done.astype(np.float32), sim.mask()
            ptr, size = (ptr + sim.n) % buffer, min(size + sim.n, buffer)
        idx_done = np.flatnonzero(done)
        if target == "mc":
            for i in idx_done:
                n_i = len(ep_a[i])
                j = (ptr + np.arange(n_i)) % buffer
                O[j], A[j], G[j] = np.array(ep_o[i]), np.array(ep_a[i]), float(win[i])
                ptr, size = (ptr + n_i) % buffer, min(size + n_i, buffer)
                ep_o[i], ep_a[i] = [], []
        if len(idx_done):
            finished += len(idx_done)
            pos, b, t = new_start(len(idx_done))
            sim.pos[idx_done], sim.b[idx_done], sim.t[idx_done] = pos, b, t
            sim.done[idx_done] = False
            sim.success[idx_done] = False
        for _ in range(updates_per_step if size >= batch * 4 else 0):
            j = rng.integers(0, size, batch)
            qa = model(torch.as_tensor(O[j])).gather(1, torch.as_tensor(A[j])[:, None]).squeeze(1)
            y = torch.as_tensor(G[j])
            if target == "td":
                with torch.no_grad():
                    o2 = torch.as_tensor(O2[j])
                    m2 = torch.as_tensor(M2[j])
                    a2 = model(o2).masked_fill(~m2, -1e9).argmax(1, keepdim=True)
                    q2 = tgt(o2).gather(1, a2).squeeze(1)
                    y = y + (1 - torch.as_tensor(D[j])) * q2
            loss = F.mse_loss(qa, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            n_upd += 1
            if n_upd % target_update == 0:
                tgt.load_state_dict(model.state_dict())
        if eval_starts is not None and finished >= next_eval:
            next_eval += eval_every
            sr = run_policy(make_dqn_policy(model), eval_seq, cfg, eval_starts).mean()
            history.append((finished, float(sr)))
            if log:
                log(f"  episode {finished:,}  eps {eps:.3f}  驗證成功率 {sr:.2%}（{time.time() - t0:.0f}s）")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "cfg": asdict(cfg), "hidden": tuple(hidden),
                    "target": target, "explore_starts": explore_starts, "dueling": dueling},
                   os.path.join(out_dir, "model.pt"))
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f)
    return model, history


def load_dqn(path):
    from .compat import torch_load
    ck = torch_load(path)
    cfg = BankrollConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["cfg"].items()})
    model = BankrollQ(obs_dim(cfg), len(cfg.bets), tuple(ck["hidden"]), ck.get("dueling", False))
    sd = ck["state_dict"]
    if any(k.startswith("net.") for k in sd):                  # 相容 dueling 之前的舊結構
        n_lin = len({k.split(".")[1] for k in sd})
        last = str(2 * (n_lin - 1))
        sd = {("out." + k.split(".", 2)[2]) if k.split(".")[1] == last else k.replace("net.", "trunk.", 1): v
              for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model, cfg


# =============================================================================
# 評估與圖表
# =============================================================================
def _summ(ep: dict, target: float) -> dict:
    f, m = ep["final"], ep["mdd"]
    q = [5, 25, 50, 75, 95]
    hist_edges = np.linspace(0, target * 1.2, 25)
    return {"success": float(ep["success"].mean()), "ruin": float(ep["ruin"].mean()),
            "timeout": float(ep["timeout"].mean()),
            "final_mean": float(f.mean()), "final_q": dict(zip(map(str, q), np.percentile(f, q).tolist())),
            "final_hist": [np.histogram(np.clip(f, 0, hist_edges[-1]), hist_edges)[0].tolist(), hist_edges.tolist()],
            "mdd_mean": float(m.mean()), "mdd_median": float(np.median(m)), "mdd_p95": float(np.percentile(m, 95)),
            "mdd_frac_mean": float(ep["mdd_frac"].mean()), "mdd_frac_p95": float(np.percentile(ep["mdd_frac"], 95)),
            "length_mean": float(ep["length"].mean()), "episodes": int(len(f))}


def _ci(x, groups, n_boot, rng):
    """平均的 95% CI：groups 為 None 時用常態近似；否則以群組為單位 bootstrap（處理重疊起點的相關）。"""
    if groups is None:
        se = x.std(ddof=1) / np.sqrt(len(x))
        return [float(x.mean() - 1.96 * se), float(x.mean() + 1.96 * se)]
    ug, inv = np.unique(groups, return_inverse=True)
    sums, cnts = np.bincount(inv, x), np.bincount(inv)
    idx = rng.integers(0, len(ug), (n_boot, len(ug)))
    st = sums[idx].sum(1) / cnts[idx].sum(1)
    return [float(np.quantile(st, 0.025)), float(np.quantile(st, 0.975))]


def compare_policies(policies: dict, seq, cfg, n_episodes: int = None, seed: int = 0, ref: str = REF,
                     starts: str = "disjoint", n_boot: int = 1000) -> dict:
    """所有策略使用相同的起點（同一段牌序），並以 ref 為基準做配對比較。

    starts="disjoint"：互不重疊的起點（預設），episode 之間不共用牌局，信賴區間用常態近似
    starts="random"  ：隨機起點（可取更多 episode，但彼此共用牌局），信賴區間以
                       「起點所在的 horizon 長區塊」為單位做 bootstrap
    """
    rng = np.random.default_rng(seed)
    if starts == "disjoint":
        st, groups = disjoint_starts(seq, cfg, n_episodes), None
    else:
        st = random_starts(seq, cfg, n_episodes, rng)
        groups = st // cfg.horizon
    eps = {name: run_episodes(p, seq, cfg, st) for name, p in policies.items()}
    out = {"_meta": {"episodes": int(len(st)), "starts": starts, "start": cfg.start, "target": cfg.target,
                     "horizon": cfg.horizon}}
    for name, ep in eps.items():
        r = _summ(ep, cfg.target)
        r["ci95"] = _ci(ep["success"].astype(float), groups, n_boot, rng)
        r["ruin_ci95"] = _ci(ep["ruin"].astype(float), groups, n_boot, rng)
        d = ep["success"].astype(float) - eps[ref]["success"].astype(float)
        r["diff_vs_ref"] = float(d.mean())
        r["diff_ci95"] = _ci(d, groups, n_boot, rng)
        out[name] = r
    return out


def sensitivity_grid(policies: dict, seq, base_cfg: BankrollConfig, start_list, horizon_list,
                     n_episodes: int = None, ref: str = REF) -> dict:
    """起始資金 × 期限的網格（目標固定為 base_cfg.target）。
    DP 需以 max(horizon_list) 求解、DQN 需以不小於 max(horizon_list) 的期限訓練（exploring starts
    會涵蓋所有較短的剩餘局數與各種資金），同一組策略即可評估整個網格。"""
    grid = {}
    for s0 in start_list:
        for h in horizon_list:
            cfg = BankrollConfig(**{**asdict(base_cfg), "start": float(s0), "horizon": int(h)})
            res = compare_policies(policies, seq, cfg, n_episodes, ref=ref)
            grid[f"{float(s0)}|{int(h)}"] = {n: {"success": r["success"], "ci95": r["ci95"], "ruin": r["ruin"],
                                     "diff_vs_ref": r["diff_vs_ref"], "diff_ci95": r["diff_ci95"]}
                                 for n, r in res.items() if not n.startswith("_")}
    return {"start_list": list(start_list), "horizon_list": list(horizon_list), "grid": grid}


def format_comparison(res: dict) -> str:
    m = res["_meta"]
    L = [f"{m['horizon']} 局內從 {m['start']:g} 到 {m['target']:g}；{m['episodes']:,} 個 episode"
         f"（{'不重疊起點' if m['starts'] == 'disjoint' else '隨機起點、區塊 bootstrap'}，所有策略同牌序）",
         f"{'策略':<30}{'成功':>8}{'破產':>8}{'到期':>8}   成功 95% CI          相對目前規則（配對）",
         ]
    for n, r in res.items():
        if n.startswith("_"):
            continue
        L.append(f"{n:<30}{r['success']:>8.2%}{r['ruin']:>8.2%}{r['timeout']:>8.2%}   "
                 f"[{r['ci95'][0]:.2%}, {r['ci95'][1]:.2%}]   {r['diff_vs_ref'] * 100:+.2f} pt "
                 f"[{r['diff_ci95'][0] * 100:+.2f}, {r['diff_ci95'][1] * 100:+.2f}]")
    L.append("")
    L.append(f"{'策略':<30}{'期末資金 5/25/50/75/95 分位':>32}{'平均':>8}   最大回撤 平均/中位/95%（單位）  比例平均/95%")
    for n, r in res.items():
        if n.startswith("_"):
            continue
        q = r["final_q"]
        L.append(f"{n:<30}{q['5']:>7.1f}{q['25']:>7.1f}{q['50']:>7.1f}{q['75']:>7.1f}{q['95']:>7.1f}"
                 f"{r['final_mean']:>8.1f}   {r['mdd_mean']:>6.1f} / {r['mdd_median']:>5.1f} / {r['mdd_p95']:>5.1f}"
                 f"          {r['mdd_frac_mean']:.0%} / {r['mdd_frac_p95']:.0%}")
    return "\n".join(L)


class _GridState:
    """熱圖用：把資金 × 剩餘局數的網格包裝成模擬器介面。"""

    def __init__(self, cfg, B, TL, v):
        self.cfg, self.bets, self.n = cfg, np.array(cfg.bets, float), B.size
        self.b, self.t = B.ravel().astype(float), cfg.horizon - TL.ravel()
        self._v = v

    def edge(self):
        return np.full(self.n, self._v)

    def obs(self):
        return make_obs(self.cfg, self.b, self.cfg.horizon - self.t, self.edge(), np.zeros((self.n, 10)))

    def mask(self):
        return self.bets[None, :] <= self.b[:, None] + 1e-9


def heatmap(policies: dict, cfg, v_values, out_path):
    # 不經過 pyplot，避免在 notebook 中切換繪圖後端
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    bgrid = np.arange(int(cfg.min_bet * 2), int(cfg.target * 2)) / 2.0
    tgrid = np.arange(1, cfg.horizon + 1)
    B, TL = np.meshgrid(bgrid, tgrid, indexing="ij")
    fig = Figure(figsize=(4 * len(v_values), 3.2 * len(policies)))
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(policies), len(v_values), squeeze=False)
    im = None
    for r, (name, pol) in enumerate(policies.items()):
        for c, v in enumerate(v_values):
            s = _GridState(cfg, B, TL, v)
            a = pol(s)
            a = np.where(s.mask()[np.arange(s.n), a], a, 0)
            ax = axes[r, c]
            im = ax.imshow(s.bets[a].reshape(B.shape), origin="lower", aspect="auto", cmap="viridis",
                           extent=[1, cfg.horizon, bgrid[0], bgrid[-1]],
                           vmin=min(cfg.bets), vmax=max(cfg.bets))
            ax.set_title(f"{name}  edge={v * 100:+.1f}%", fontsize=9)
            ax.set_xlabel("rounds left"); ax.set_ylabel("bankroll (units)")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="bet (units)")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    return out_path


# =============================================================================
# 命令列
# =============================================================================
def _cfg_from(a):
    return BankrollConfig(start=a.start, target=a.target, horizon=a.horizon,
                          bets=tuple(int(x) for x in a.bets.split(",")), edge_input=a.edge_input)


def main():
    from .cli import add_rule_args, obs_from_args, rules_from_args
    p = argparse.ArgumentParser(description="資金管理（期限內翻倍）")
    sub = p.add_subparsers(dest="cmd")
    sub.required = True

    def bank_args(q):
        q.add_argument("--start", type=float, default=50)
        q.add_argument("--target", type=float, default=100)
        q.add_argument("--horizon", type=int, default=500)
        q.add_argument("--bets", default="1,2,4,8,16")
        q.add_argument("--edge-input", default="v", choices=["v", "shoe", "both"])

    q = sub.add_parser("record", help="錄製序列")
    add_rule_args(q)
    q.add_argument("--play", default="basic", help="出牌策略：basic 或 DQN 模型（.pt）")
    q.add_argument("--edge", required=True, help="優勢預測：兩頭 DQN 模型（.pt）或線性下注模型（.npz）")
    q.add_argument("--rounds", type=int, default=3_000_000)
    q.add_argument("--seed", type=int, default=11)
    q.add_argument("--out", required=True)

    q = sub.add_parser("dp", help="DP 上限參照")
    bank_args(q)
    q.add_argument("--seq", required=True)
    q.add_argument("--buckets", type=int, default=12)
    q.add_argument("--out", required=True)

    q = sub.add_parser("train", help="訓練資金管理 DQN")
    bank_args(q)
    q.add_argument("--seq", required=True)
    q.add_argument("--eval-seq")
    q.add_argument("--episodes", type=int, default=300_000)
    q.add_argument("--n-envs", type=int, default=512)
    q.add_argument("--method", default="td", choices=["td", "mc"],
                   help="td：Double DQN（預設）；mc：Monte Carlo 回報")
    q.add_argument("--explore-starts", type=float, default=0.5,
                   help="從隨機資金與剩餘局數開始的 episode 比例（只在訓練時）")
    q.add_argument("--no-dueling", action="store_true")
    q.add_argument("--updates-per-step", type=int, default=4)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--out", required=True)

    q = sub.add_parser("evaluate", help="比較各策略（結局、期末資金、最大回撤、敏感度、訓練 vs 測試）")
    bank_args(q)
    q.add_argument("--seq", required=True, help="測試序列")
    q.add_argument("--train-seq", help="也在訓練序列上評估，比較兩者差距（過擬合程度）")
    q.add_argument("--dqn", nargs="*", default=[])
    q.add_argument("--dp")
    q.add_argument("--episodes", type=int, default=None, help="最多幾個 episode（不重疊起點時受序列長度限制）")
    q.add_argument("--starts", default="disjoint", choices=["disjoint", "random"])
    q.add_argument("--grid-starts", default=None, help="敏感度網格的起始資金，例如 25,50,75")
    q.add_argument("--grid-horizons", default=None, help="敏感度網格的期限，例如 250,500,1000")
    q.add_argument("--seed", type=int, default=2027)
    q.add_argument("--json")

    q = sub.add_parser("heatmap", help="下注熱圖")
    bank_args(q)
    q.add_argument("--dqn")
    q.add_argument("--dp")
    q.add_argument("--out", default="bankroll_heatmap.png")
    a = p.parse_args()

    if a.cmd == "record":
        from .agent import DQNAgent
        from .strategies import BasicStrategy
        rules, obs_cfg = rules_from_args(a), obs_from_args(a)
        play = BasicStrategy()
        if a.play != "basic":
            play = DQNAgent.load(a.play)
            obs_cfg = play.obs_cfg
        if a.edge.endswith(".npz"):
            from .bet_model import LinearBetModel
            edge = _EdgeFromLinear(LinearBetModel.load(a.edge, rules))
        else:
            ag = DQNAgent.load(a.edge)
            obs_cfg = ag.obs_cfg
            edge = _EdgeFromAgent(ag)
        d = record_sequence(rules, obs_cfg, play, edge, a.rounds, a.seed)
        np.savez_compressed(a.out, **d)
        print(f"已存 {a.out}：{a.rounds:,} 局；每單位平均結果 {d['R'].mean() * 100:+.3f}%，"
              f"預測優勢為正的局 {np.mean(d['V'] > 0):.1%}")
    elif a.cmd == "dp":
        dp = solve_dp(load_seq(a.seq), _cfg_from(a), a.buckets)
        np.savez_compressed(a.out, **dp)
    elif a.cmd == "train":
        train_dqn(load_seq(a.seq), _cfg_from(a), a.episodes, a.n_envs, target=a.method,
                  explore_starts=a.explore_starts, dueling=not a.no_dueling,
                  updates_per_step=a.updates_per_step,
                  eval_seq=load_seq(a.eval_seq) if a.eval_seq else None, seed=a.seed, out_dir=a.out)
    elif a.cmd == "evaluate":
        cfg = _cfg_from(a)
        pols = dict(BASELINES)
        if a.dp:
            pols["DP 最佳解（上限參照）"] = make_dp_policy(dict(np.load(a.dp)))
        for path in a.dqn:
            m, c = load_dqn(path)
            pols[f"DQN（{os.path.basename(os.path.dirname(path)) or path}）"] = make_dqn_policy(m, c)
        out = {}
        for tag, path in [("test", a.seq), ("train", a.train_seq)]:
            if not path:
                continue
            res = compare_policies(pols, load_seq(path), cfg, a.episodes, a.seed, starts=a.starts)
            out[tag] = res
            print(f"\n==== {'測試序列（未見過的牌局，泛化表現）' if tag == 'test' else '訓練序列（DQN 與 DP 都從這裡學習）'} ====")
            print(format_comparison(res))
        if "train" in out:
            print("\n==== 訓練序列 − 測試序列（成功機率，正值代表在見過的序列上表現較好）====")
            for n in pols:
                print(f"{n:<30}{(out['train'][n]['success'] - out['test'][n]['success']) * 100:+.2f} pt")
        if a.grid_starts and a.grid_horizons:
            gs = [float(x) for x in a.grid_starts.split(",")]
            gh = [int(x) for x in a.grid_horizons.split(",")]
            g = sensitivity_grid(pols, load_seq(a.seq), cfg, gs, gh, a.episodes)
            out["grid"] = g
            print(f"\n==== 敏感度網格（目標 {cfg.target:g}，測試序列）：成功機率 ====")
            for n in pols:
                print(f"[{n}]")
                print("  起始資金＼期限 " + "".join(f"{h:>10}" for h in gh))
                for s0 in gs:
                    print(f"  {s0:>12g}   " + "".join(f"{g['grid'][f'{s0}|{h}'][n]['success']:>10.2%}" for h in gh))
        if a.json:
            with open(a.json, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
    elif a.cmd == "heatmap":
        cfg = _cfg_from(a)
        pols = {}
        if a.dqn:
            m, c = load_dqn(a.dqn)
            pols["DQN"] = make_dqn_policy(m, c)
        if a.dp:
            pols["DP"] = make_dp_policy(dict(np.load(a.dp)))
        print("已存", heatmap(pols, cfg, [-0.02, 0.0, 0.02], a.out))


if __name__ == "__main__":
    main()
