"""Double DQN（PyTorch）。

目標值（TrainConfig.target_mode）：
  - "td" : Double DQN，y = r + gamma * Q_target(s', argmax_a' Q(s', a'))
  - "mc" : Monte Carlo，y = 該局實際總報酬 G。gamma=1 且只有終局獎勵，所以每一步的 G 都等於整局報酬。
           沒有 bootstrap 偏差，但 G 反映的是「行為策略」（含 ε 探索）的價值。
下注頭（TrainConfig.bet_head，開啟下注時）：
  - "regress"（預設）：獨立的下注頭（BetHead），不共用主幹。目標是每局實際的每單位報酬
           （期望結算），只用整局出牌皆為貪婪選擇的局；線性部分以遞迴最小平方法精確求解，
           MLP 學殘差。這樣下注頭不會繼承出牌頭的高估偏差。
  - "value"：下注頭只輸出 V(s)，下 b 倍的 Q 值 = b × V(s)。訓練時所有報酬換算成
           「每單位下注」，所以 V 的 TD 目標就是發牌後出牌頭的最佳 Q 值；下注不影響發牌，
           不需要探索下注，每一局都能用來學 V。
  - "q"：舊做法，每個下注倍數各一個 Q 值。
  - "rollout" : 在決策點複製環境，每個合法動作各跑一次、之後照目前的貪婪策略打完；
           所有分支看到相同的後續牌序（common random numbers），一次得到所有動作的目標值。

TrainConfig.expected_settlement=True 時，局結束的報酬改用「莊家補牌的期望結果」，
只影響訓練目標，評估一律使用實際報酬。

相對原專案的修正：
  - gamma = 1.0（只有終局獎勵）
  - 有 target network，用 Double DQN 計算目標值
  - 輸出層為線性（原專案 Q 值被 BatchNorm + LeakyReLU 壓縮）
  - 非法動作以 -inf 遮罩，下一狀態的 max 也只在合法動作中取
  - Huber loss + 梯度裁剪；reward 依最大下注倍數縮放以穩定訓練
"""
import copy

from .config import N_PLAY_ACTIONS as N_PLAY_ACTIONS_
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .compat import huber, torch_load
from .config import ObsConfig, RuleConfig, TrainConfig


class QNetwork(nn.Module):
    """MLP Q 網路（共用主幹 + 出牌頭，可選下注頭）。

    dueling=True 時出牌頭 Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)。
    n_bet > 0 時另開下注頭，只輸出一個數值 V_bet(s)（每單位下注的期望報酬），
    並複製成 n_bet 個輸出，使輸出維度與動作數一致（下注倍數由 agent 乘上）。"""

    def __init__(self, obs_dim: int, n_actions: int, hidden=(256, 256), dueling=False, n_bet=0):
        super().__init__()
        self.n_bet = n_bet
        n_actions = n_actions - n_bet
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.dueling = dueling
        if dueling:
            self.v = nn.Linear(d, 1)
            self.a = nn.Linear(d, n_actions)
        else:
            self.out = nn.Linear(d, n_actions)       # 線性輸出

        if n_bet:
            self.bet_v = nn.Linear(d, 1)

    def forward(self, x):
        h = self.trunk(x)
        if self.dueling:
            a = self.a(h)
            q = self.v(h) + a - a.mean(1, keepdim=True)
        else:
            q = self.out(h)
        if self.n_bet:
            q = torch.cat([q, self.bet_v(h).expand(-1, self.n_bet)], dim=1)
        return q


class BetHead(nn.Module):
    """獨立的下注頭（regress 模式）：V(s) = 線性(特徵) + MLP 殘差(特徵)。

    - 與出牌網路完全分開，不共用主幹，梯度互不干擾
    - 輸入是從 state 的牌堆特徵換算出的「組成偏差」（與線性下注模型相同的歸納偏差）：
        seen_counts：[已出現總數/52, 各點數比平均多出現的張數/52（以 8 點為參照）]
        hilo       ：[running count]
        none       ：無特徵（只學平均值）
    - 線性部分以遞迴最小平方法線上精確求解，含遺忘因子，讓較舊（出牌較差時）的資料逐漸淡出
    - MLP 只學線性部分的殘差，輸出層初始化為 0，一開始完全等於線性模型
    """

    def __init__(self, obs_cfg, pdim: int, hidden: int = 16,
                 forget: float = 1 - 1 / 200_000, ridge: float = 1e-3):
        super().__init__()
        from .env import _SEEN_SCALE
        from .shoe import CARDS_PER_DECK
        self.mode, self.pdim = obs_cfg.shoe_info, pdim
        self.nf = {"seen_counts": 10, "hilo": 1, "none": 0}[self.mode]
        self.register_buffer("w", torch.zeros(self.nf + 1))          # [截距, 係數...]
        self.register_buffer("scale", torch.as_tensor(_SEEN_SCALE, dtype=torch.float32))
        self.register_buffer("share", torch.as_tensor(CARDS_PER_DECK / 52.0, dtype=torch.float32))
        self.keep = [i for i in range(10) if i != 7]
        self.mlp = None
        if hidden and self.nf:
            self.mlp = nn.Sequential(nn.Linear(self.nf, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
        self.forget, self.ridge = forget, ridge
        k = self.nf + 1
        self.XtX = np.zeros((k, k))
        self.Xty = np.zeros(k)
        self.n_obs = 0.0

    def features(self, x):
        shoe = x[:, self.pdim:]
        if self.mode == "seen_counts":
            seen = shoe * self.scale
            S = seen.sum(1, keepdim=True)
            f = (seen - S * self.share) / 52.0
            return torch.cat([S / 52.0, f[:, self.keep]], dim=1)
        if self.mode == "hilo":
            return shoe
        return x[:, :0]

    def linear(self, f):
        X = torch.cat([torch.ones(len(f), 1, device=f.device), f], dim=1)
        return X @ self.w

    def forward(self, x):
        f = self.features(x)
        v = self.linear(f)
        if self.mlp is not None:
            v = v + self.mlp(f).squeeze(1)
        return v[:, None]

    @torch.no_grad()
    def observe(self, obs: np.ndarray, y: np.ndarray):
        """加入新資料到遞迴最小平方的累積量（批次版遺忘）。"""
        f = self.features(torch.as_tensor(obs, dtype=torch.float32, device=self.w.device)).cpu().numpy()
        X = np.hstack([np.ones((len(f), 1)), f])
        lam = self.forget ** len(y)
        self.XtX = lam * self.XtX + X.T @ X
        self.Xty = lam * self.Xty + X.T @ y
        self.n_obs = lam * self.n_obs + len(y)

    def refit(self):
        k = self.nf + 1
        w = np.linalg.solve(self.XtX + self.ridge * np.eye(k), self.Xty)
        self.w.copy_(torch.as_tensor(w, dtype=torch.float32))


class BetBuffer:
    """下注頭用：每局一筆（下注時的 state、該局每單位下注的報酬）。"""

    def __init__(self, cap: int, obs_dim: int):
        self.cap, self.n, self.i = cap, 0, 0
        self.obs = np.zeros((cap, obs_dim), np.float32)
        self.y = np.zeros(cap, np.float32)

    def push(self, o, y):
        self.obs[self.i], self.y[self.i] = o, y
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, bs, rng):
        idx = rng.integers(0, self.n, bs)
        return self.obs[idx], self.y[idx]


class RolloutBuffer:
    """rollout 模式用：每筆存一個 state 與所有合法動作的目標值。"""

    def __init__(self, cap: int, obs_dim: int, n_actions: int):
        self.cap, self.n, self.i = cap, 0, 0
        self.obs = np.zeros((cap, obs_dim), np.float32)
        self.tgt = np.zeros((cap, n_actions), np.float32)
        self.tmask = np.zeros((cap, n_actions), bool)

    def push(self, o, t, m):
        j = self.i
        self.obs[j], self.tgt[j], self.tmask[j] = o, t, m
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, bs: int, rng: np.random.Generator):
        idx = rng.integers(0, self.n, bs)
        return self.obs[idx], self.tgt[idx], self.tmask[idx]


class ReplayBuffer:
    def __init__(self, cap: int, obs_dim: int, n_actions: int):
        self.cap, self.n, self.i = cap, 0, 0
        self.obs = np.zeros((cap, obs_dim), np.float32)
        self.next_obs = np.zeros((cap, obs_dim), np.float32)
        self.act = np.zeros(cap, np.int64)
        self.rew = np.zeros(cap, np.float32)
        self.done = np.zeros(cap, np.float32)
        self.next_mask = np.zeros((cap, n_actions), bool)
        self.ret = np.zeros(cap, np.float32)          # Monte Carlo 回報

    def push(self, o, a, r, o2, d, m2, g=0.0):
        j = self.i
        self.obs[j], self.act[j], self.rew[j] = o, a, r
        self.next_obs[j], self.done[j], self.next_mask[j] = o2, d, m2
        self.ret[j] = g
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, bs: int, rng: np.random.Generator):
        idx = rng.integers(0, self.n, bs)
        return (self.obs[idx], self.act[idx], self.rew[idx],
                self.next_obs[idx], self.done[idx], self.next_mask[idx], self.ret[idx])


class DQNAgent:
    name = "dqn"

    def __init__(self, obs_dim: int, n_actions: int, rules: RuleConfig,
                 obs_cfg: ObsConfig, cfg: TrainConfig):
        self.rules, self.obs_cfg, self.cfg = rules, obs_cfg, cfg
        self.obs_dim, self.n_actions = obs_dim, n_actions
        self.device = torch.device(cfg.device)
        from .config import N_PLAY_ACTIONS
        # 下注頭模式："value"（單一價值 × 倍數）、"q"（每個倍數各一個 Q）、None（未開啟下注）
        self.bet_mode = (cfg.bet_head if rules.enable_betting else None)
        n_bet = rules.n_bet_actions if self.bet_mode == "value" else 0
        # regress 模式：出牌網路只輸出 5 個出牌動作，下注由獨立的 BetHead 負責
        self.q_out = N_PLAY_ACTIONS if self.bet_mode == "regress" else n_actions
        self.q = QNetwork(obs_dim, self.q_out, cfg.hidden, cfg.dueling, n_bet).to(self.device)
        self.q_target = QNetwork(obs_dim, self.q_out, cfg.hidden, cfg.dueling, n_bet).to(self.device)
        # 出牌先驗：residual 模式下 Q = 網路 + basic strategy 先驗，網路出牌頭初始為 0
        self.prior = None
        if cfg.play_prior == "residual":
            from .prior import BasicPrior
            self.prior = BasicPrior(rules, obs_cfg, cfg.prior_margin).to(self.device)
            for net in (self.q, self.q_target):
                heads = [net.v, net.a] if net.dueling else [net.out]
                for h in heads:
                    nn.init.zeros_(h.weight)
                    nn.init.zeros_(h.bias)
        self.demo = None                   # 老師示範資料（DP 蒸餾時設定），供 RL 階段的間距損失使用
        self._demo_rng = np.random.default_rng(cfg.seed + 4242)
        self.anchor = None
        if cfg.anchor_weight > 0 and cfg.teacher == "basic":
            from .prior import BasicPrior
            self.anchor = BasicPrior(rules, obs_cfg, 1.0).to(self.device)
        self.bet_head = None
        if self.bet_mode == "regress":
            from .env import play_feature_dim
            assert cfg.target_mode != "rollout", "regress 下注頭不支援 rollout 目標"
            self.bet_head = BetHead(obs_cfg, play_feature_dim(obs_cfg), cfg.bet_hidden,
                                    1 - 1 / cfg.bet_memory).to(self.device)
            self.bet_sizes_t = torch.as_tensor(rules.bet_sizes, dtype=torch.float32, device=self.device)
            self.bet_buf = BetBuffer(cfg.buffer_size, obs_dim)
            self.bet_opt = (torch.optim.Adam(self.bet_head.mlp.parameters(), lr=cfg.bet_lr,
                                             weight_decay=1e-4) if self.bet_head.mlp else None)
        # 選動作用的倍數：出牌動作 ×1，下注動作 × 下注倍數（value 模式）
        mult = np.ones(n_actions, np.float32)
        if self.bet_mode == "value":
            mult[N_PLAY_ACTIONS:] = rules.bet_sizes
            assert cfg.target_mode != "rollout", "value 下注頭不支援 rollout 目標"
        self.bet_mult = mult
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        # 學習率由 lr 線性衰減到 lr_end（以梯度更新次數計）
        n_total = max(1, (cfg.total_steps - cfg.learning_starts) // max(1, cfg.train_every))
        end_ratio = cfg.lr_end / cfg.lr
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda u: 1.0 + min(1.0, u / n_total) * (end_ratio - 1.0))
        # value / regress 模式下報酬已換算成每單位下注，不需再縮放
        self.reward_scale = 1.0 / max(rules.bet_sizes) if self.bet_mode == "q" else 1.0
        self.n_updates = 0

    def _selection_bonus(self, x):
        """決策門檻：basic 動作在選擇時加上 deviate_threshold（不影響訓練）。"""
        tau = getattr(self.cfg, "deviate_threshold", 0.0)
        if tau <= 0:
            return None
        if getattr(self, "_sel_prior", None) is None or self._sel_prior.margin != tau:
            from .prior import BasicPrior
            self._sel_prior = BasicPrior(self.rules, self.obs_cfg, tau).to(self.device)
        return self._sel_prior(x)

    def _margin_loss(self, q, a_exp, m, margin):
        """DQfD 間距損失：max_a [Q(s,a) + margin·1(a≠a_E)] − Q(s,a_E)，只在合法動作中取最大。"""
        q_e = q.gather(1, a_exp[:, None]).squeeze(1)
        l = torch.full_like(q, margin)
        l.scatter_(1, a_exp[:, None], 0.0)
        big = (q + l).masked_fill(~m, -1e9).max(1).values
        return (big - q_e).mean()

    def _anchor_loss(self, o, q_all):
        """RL 階段的模仿學習損失（只作用於出牌階段的 state）。"""
        a_exp = self.anchor.actions(o)
        rows = a_exp >= 0
        if not bool(rows.any()):
            return q_all.sum() * 0.0
        q = q_all[rows, :N_PLAY_ACTIONS_]
        return self._margin_loss(q, a_exp[rows], self.anchor.legal_from_obs(o[rows]),
                                 self.cfg.anchor_margin)

    def _demo_loss(self):
        """DQfD：從老師示範資料抽一批，要求老師的動作排在最前面。"""
        d = self.demo
        idx = self._demo_rng.integers(0, len(d["act"]), self.cfg.batch_size)
        o = torch.as_tensor(d["obs"][idx], device=self.device)
        q = self._q(self.q, o)[:, :N_PLAY_ACTIONS_]
        return self._margin_loss(q, torch.as_tensor(d["act"][idx], device=self.device),
                                 torch.as_tensor(d["mask"][idx], device=self.device),
                                 self.cfg.anchor_margin)

    def distill_step(self, o, m, a_exp, ev, lam: float = 1.0, margin: float = 0.05) -> float:
        """DP 蒸餾：每個合法動作的 Q 回歸到 DP 期望值，並要求 DP 選的動作高出 margin。"""
        o, m, a_exp, ev = (torch.as_tensor(x, device=self.device) for x in (o, m, a_exp, ev))
        q = self._q(self.q, o)[:, :N_PLAY_ACTIONS_]
        known = m & torch.isfinite(ev)
        loss = self._loss(q[known], ev[known]) + lam * self._margin_loss(q, a_exp, m, margin)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.opt.step()
        return loss.item()

    def _loss(self, x, y, beta=None):
        """MSE（期望值的無偏估計）或 Huber（舊版）。"""
        if self.cfg.loss == "mse":
            return F.mse_loss(x, y)
        return huber(x, y, beta or self.cfg.huber_beta)

    def _q(self, net, x):
        """網路輸出（出牌頭加上 basic strategy 先驗，若有）。"""
        q = net(x)
        if self.prior is not None:
            p = self.prior(x)
            q = torch.cat([q[:, :p.shape[1]] + p, q[:, p.shape[1]:]], dim=1)
        return q

    # ------------------------------------------------------------------ 選動作
    @torch.no_grad()
    def q_values(self, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        qp = self._q(self.q, x)
        bonus = self._selection_bonus(x)
        if bonus is not None:
            qp = torch.cat([qp[:, :bonus.shape[1]] + bonus, qp[:, bonus.shape[1]:]], dim=1)
        if self.bet_mode == "regress":
            q = torch.cat([qp, self.bet_head(x) * self.bet_sizes_t], dim=1).cpu().numpy()
        else:
            q = qp.cpu().numpy() * self.bet_mult     # 下注動作 = 倍數 × V(s)
        return np.where(masks, q, -np.inf)

    def act_batch(self, obs, masks, eps: float, rng: np.random.Generator,
                  return_explore: bool = False):
        acts = self.q_values(obs, masks).argmax(1)
        explore = rng.random(len(obs)) < eps
        for i in np.flatnonzero(explore):
            acts[i] = rng.choice(np.flatnonzero(masks[i]))
        return (acts, explore) if return_explore else acts

    def act(self, env, obs):
        """評估介面：貪婪策略。

        只學出牌的模型（訓練時關閉下注，n_actions = 5）也能用在有下注的環境：
        下注階段固定下最小注，出牌階段只看前 5 個動作的 mask。"""
        from .config import N_PLAY_ACTIONS
        mask = env.legal_mask()
        if env.phase == "bet" and self.n_actions <= N_PLAY_ACTIONS:
            return N_PLAY_ACTIONS
        m = mask[:self.n_actions]
        return int(self.q_values(obs[None], m[None])[0].argmax())

    def first_decision_action(self, cards, dealer_up, mask) -> int:
        """剛洗好牌堆時、起手牌的決策（策略表用）。"""
        from .env import encode_obs
        x = encode_obs(self.rules, self.obs_cfg, phase="play", hand_cards=cards,
                       dealer_up=dealer_up, mask=mask, seen_counts=np.zeros(10), running_count=0)
        return int(self.q_values(x[None], mask[None])[0].argmax())

    # ------------------------------------------------------------------ 更新
    def update(self, batch) -> float:
        o, a, r, o2, d, m2, g = (torch.as_tensor(x, device=self.device) for x in batch)
        assert int(a.max()) < self.q_out, "regress 模式下，下注轉移不應放入出牌的 replay buffer"
        r = r * self.reward_scale
        q_all = self._q(self.q, o)
        q = q_all.gather(1, a[:, None]).squeeze(1)
        if self.cfg.target_mode == "mc":
            y = g * self.reward_scale
        else:
            y = self._td_target(r, o2, d, m2)
        loss = self._loss(q, y)
        if self.cfg.anchor_weight > 0:
            if self.demo is not None:
                loss = loss + self.cfg.anchor_weight * self._demo_loss()
            elif self.anchor is not None:
                loss = loss + self.cfg.anchor_weight * self._anchor_loss(o, q_all)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.opt.step()
        self.sched.step()
        self.n_updates += 1
        if self.cfg.target_mode == "td" and self.n_updates % self.cfg.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return loss.item()

    def pretrain_step(self, o, a_exp, m, g, margin: float = 0.05, lam: float = 1.0) -> float:
        """模仿學習（DQfD 式）：專家動作的 Q 回歸到實際回報，並要求專家動作比其他合法動作
        高出 margin：loss = Huber(Q(s,a_E), G) + lam * [max_a (Q(s,a) + margin·1[a≠a_E]) − Q(s,a_E)]"""
        o, a_exp, m, g = (torch.as_tensor(x, device=self.device) for x in (o, a_exp, m, g))
        m = m[:, :self.q_out]
        q = self._q(self.q, o)[:, :self.q_out]
        q_e = q.gather(1, a_exp[:, None]).squeeze(1)
        loss = self._loss(q_e, g) + lam * self._margin_loss(q, a_exp, m, margin)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.opt.step()
        return loss.item()

    def push_bet(self, obs_bet: np.ndarray, y: np.ndarray):
        """一批完成的局：下注時的 state 與每單位下注報酬。"""
        if len(y) == 0:
            return
        self.bet_head.observe(obs_bet, y)
        for o, t in zip(obs_bet, y):
            self.bet_buf.push(o, t)

    def update_bet(self, rng, n_steps: int = 1) -> float:
        """線性部分精確重解；MLP 以 Adam 學殘差。"""
        self.bet_head.refit()
        if self.bet_opt is None or self.bet_buf.n < self.cfg.bet_batch:
            return 0.0
        loss_v = 0.0
        for _ in range(n_steps):
            o, y = self.bet_buf.sample(self.cfg.bet_batch, rng)
            x = torch.as_tensor(o, device=self.device)
            f = self.bet_head.features(x)
            with torch.no_grad():
                resid = torch.as_tensor(y, device=self.device) - self.bet_head.linear(f)
            loss = self._loss(self.bet_head.mlp(f).squeeze(1), resid, beta=2.0)
            self.bet_opt.zero_grad()
            loss.backward()
            self.bet_opt.step()
            loss_v = loss.item()
        return loss_v

    def update_rollout(self, batch) -> float:
        o, t, m = (torch.as_tensor(x, device=self.device) for x in batch)
        q = self.q(o)
        loss = self._loss(q[m], t[m] * self.reward_scale)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.opt.step()
        self.sched.step()
        self.n_updates += 1
        return loss.item()

    def rollout_targets(self, env):
        """在 env 目前的決策點，對每個合法動作複製環境並照貪婪策略打完。
        所有分支共用相同的後續牌序，因此動作之間的差異不受運氣影響。"""
        mask = env.legal_mask()
        tgt = np.zeros(self.n_actions, np.float32)
        for a in np.flatnonzero(mask):
            e = copy.deepcopy(env)
            obs, r, done, info = e.step(a)
            while not done:
                obs, r, done, info = e.step(self.act(e, obs))
            tgt[a] = info.get("expected_reward", r)
        return tgt, mask

    @torch.no_grad()
    def _td_target(self, r, o2, d, m2):
        # Double DQN：online 網路選動作、target 網路估值，只考慮合法動作
        m2 = m2[:, :self.q_out]
        q_online = self._q(self.q, o2).masked_fill(~m2, -1e9)
        a2 = q_online.argmax(1, keepdim=True)
        q_next = self._q(self.q_target, o2).gather(1, a2).squeeze(1)
        q_next = torch.where(d > 0, torch.zeros_like(q_next), q_next)
        return r + self.cfg.gamma * q_next

    # ------------------------------------------------------------------ 存取
    def save(self, path: str):
        extra = {}
        if self.bet_head is not None:
            extra = {"bet_state": self.bet_head.state_dict(),
                     "bet_rls": (self.bet_head.XtX, self.bet_head.Xty, self.bet_head.n_obs)}
        torch.save({**extra, "state_dict": self.q.state_dict(), "rules": asdict(self.rules),
                    "obs_cfg": asdict(self.obs_cfg), "cfg": asdict(self.cfg),
                    "obs_dim": self.obs_dim, "n_actions": self.n_actions}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "DQNAgent":
        ck = torch_load(path, map_location=device)
        ck["obs_cfg"].setdefault("total_encoding", "scalar")    # 相容舊版模型
        ck["cfg"].setdefault("dueling", False)
        ck["cfg"].setdefault("bet_head", "q")
        ck["cfg"].setdefault("loss", "huber")                 # 舊模型以 Huber 訓練
        rules = RuleConfig(**{k: tuple(v) if isinstance(v, list) else v
                              for k, v in ck["rules"].items()})
        cfg = TrainConfig(**{k: tuple(v) if isinstance(v, list) else v
                             for k, v in ck["cfg"].items()})
        cfg.device = device
        agent = cls(ck["obs_dim"], ck["n_actions"], rules, ObsConfig(**ck["obs_cfg"]), cfg)
        agent.q.load_state_dict(ck["state_dict"])
        agent.q_target.load_state_dict(ck["state_dict"])
        if agent.bet_head is not None and "bet_state" in ck:
            agent.bet_head.load_state_dict(ck["bet_state"])
            agent.bet_head.XtX, agent.bet_head.Xty, agent.bet_head.n_obs = ck["bet_rls"]
        return agent
