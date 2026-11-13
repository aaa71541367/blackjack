"""訓練 DQN。

範例：
  python -m blackjack_rl.train --decks 4-8 --shoe-info seen_counts --steps 3000000 --out runs/seen
  python -m blackjack_rl.train --decks 6 --no-betting --shoe-info none --out runs/basic
  python -m blackjack_rl.train --decks 6 --no-betting --shoe-info none --target mc --lr-end 1e-5 --out runs/mc
"""
import argparse
import csv
import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from .agent import DQNAgent, ReplayBuffer, RolloutBuffer
from .cli import add_rule_args, obs_from_args, rules_from_args
from .config import ObsConfig, RuleConfig, TrainConfig
from .env import BlackjackEnv
from .evaluate import evaluate_policy, format_result


def linear_eps(step, cfg: TrainConfig):
    frac = min(1.0, step / max(1, cfg.eps_decay_steps))
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def pretrain_basic(agent, buf, rules, obs_cfg, cfg, rng):
    """模仿學習預訓練：用 basic strategy 打 cfg.pretrain_rounds 局，
    以 DQfD 式損失預訓練出牌頭，並把這些轉移放進 replay buffer 作為 RL 的起始資料。"""
    from .config import N_PLAY_ACTIONS
    from .strategies import BasicStrategy
    t0 = time.time()
    env = BlackjackEnv(rules, obs_cfg, seed=cfg.seed + 777, expected_settlement=cfg.expected_settlement)
    bs = BasicStrategy()
    O, A, M, G = [], [], [], []
    for _ in range(cfg.pretrain_rounds):
        obs, _ = env.reset()
        traj, done = [], False
        while not done:
            if env.phase == "bet":
                obs, r, done, info = env.step(N_PLAY_ACTIONS)   # 預訓練只學出牌，下最小注
                continue
            m = env.legal_mask()
            a = bs.act(env)
            o2, r, done, info = env.step(a)
            r_train = info.get("expected_reward", r) if done else r
            r_train /= env.base_bet
            traj.append((obs.copy(), a, r_train, o2, float(done), env.legal_mask(), m))
            obs = o2
        g = sum(t[2] for t in traj)
        for o, a, rt, o2, d, m2, m in traj:
            buf.push(o, a, rt, o2, d, m2, g=g)
            O.append(o); A.append(a); M.append(m); G.append(g)
    O, A = np.array(O, np.float32), np.array(A, np.int64)
    M, G = np.array(M, bool), np.array(G, np.float32)
    losses = []
    for _ in range(cfg.pretrain_updates):
        idx = rng.integers(0, len(A), cfg.batch_size)
        losses.append(agent.pretrain_step(O[idx], A[idx], M[idx], G[idx], cfg.pretrain_margin))
    agent.q_target.load_state_dict(agent.q.state_dict())
    print(f"模仿學習預訓練：{cfg.pretrain_rounds:,} 局、{len(A):,} 個決策、{cfg.pretrain_updates:,} 次更新，"
          f"最後 loss {np.mean(losses[-200:]):.4f}（{time.time() - t0:.0f}s）", flush=True)


def load_play_head(agent, path):
    """從已訓練模型複製出牌網路的權重（例如只做蒸餾、未開啟下注的模型）。"""
    src = DQNAgent.load(path, device=str(agent.device))
    if src.obs_dim != agent.obs_dim or src.obs_cfg != agent.obs_cfg:
        raise ValueError(f"{path} 的 state 設定與目前不同：{src.obs_cfg} vs {agent.obs_cfg}")
    try:
        agent.q.load_state_dict(src.q.state_dict())
    except RuntimeError as e:
        raise ValueError(f"{path} 的出牌網路結構與目前不同（hidden、dueling、下注頭模式需一致）") from e
    agent.q_target.load_state_dict(agent.q.state_dict())
    print(f"已從 {path} 載入出牌頭", flush=True)


def pretrain_dp(agent, buf, rules, obs_cfg, cfg, rng):
    """DP 蒸餾：用 DP 出牌收集示範資料，各動作 Q 回歸到 DP 期望值並加上間距損失；
    示範資料保留在 agent.demo，RL 階段以 --anchor-weight 持續約束。"""
    from .teacher import load_or_collect
    t0 = time.time()
    d = load_or_collect(cfg.teacher_cache or None, rules, obs_cfg, "dp", cfg.pretrain_rounds,
                        seed=cfg.seed + 777, expected=cfg.expected_settlement)
    for i in range(len(d["tr_act"])):          # 老師的轉移也作為 RL 的起始資料
        buf.push(d["tr_obs"][i], d["tr_act"][i], d["tr_rew"][i], d["tr_obs2"][i],
                 d["tr_done"][i], d["tr_mask2"][i], g=d["tr_g"][i])
    losses = []
    for _ in range(cfg.pretrain_updates):
        idx = rng.integers(0, len(d["act"]), cfg.batch_size)
        losses.append(agent.distill_step(d["obs"][idx], d["mask"][idx], d["act"][idx], d["ev"][idx],
                                         margin=cfg.pretrain_margin))
    agent.q_target.load_state_dict(agent.q.state_dict())
    agent.demo = {"obs": d["obs"], "mask": d["mask"], "act": d["act"]}
    print(f"DP 蒸餾：{len(d['act']):,} 個決策、{cfg.pretrain_updates:,} 次更新，"
          f"最後 loss {np.mean(losses[-200:]):.4f}（{time.time() - t0:.0f}s）", flush=True)


def train(rules: RuleConfig, obs_cfg: ObsConfig, cfg: TrainConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    envs = [BlackjackEnv(rules, obs_cfg, seed=cfg.seed * 1000 + i,
                         expected_settlement=cfg.expected_settlement) for i in range(cfg.n_envs)]
    obs = np.stack([e.reset()[0] for e in envs])
    masks = np.stack([e.legal_mask() for e in envs])
    agent = DQNAgent(envs[0].obs_dim, envs[0].n_actions, rules, obs_cfg, cfg)
    if cfg.init_play:
        load_play_head(agent, cfg.init_play)
    rollout = cfg.target_mode == "rollout"
    if rollout:
        buf = RolloutBuffer(cfg.buffer_size, envs[0].obs_dim, envs[0].n_actions)
    else:
        buf = ReplayBuffer(cfg.buffer_size, envs[0].obs_dim, envs[0].n_actions)

    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump({"rules": asdict(rules), "obs": asdict(obs_cfg), "train": asdict(cfg)},
                  f, indent=2, ensure_ascii=False)
    log_f = open(os.path.join(cfg.out_dir, "log.csv"), "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["step", "epsilon", "train_reward_recent", "loss", "eval_reward", "eval_ci_lo",
                  "eval_ci_hi", "eval_ev_per_unit", "eval_mean_bet"])

    if cfg.pretrain_rounds and not rollout:
        if cfg.teacher == "dp":
            pretrain_dp(agent, buf, rules, obs_cfg, cfg, rng)
        else:
            pretrain_basic(agent, buf, rules, obs_cfg, cfg, rng)
    pending = [[] for _ in envs]          # 每個環境目前這局的 transition，局結束才放入 buffer
    regress = agent.bet_mode == "regress"
    bet_obs = [None] * len(envs)          # regress：這局下注時的 state
    explored = [False] * len(envs)        # regress：這局出牌是否有探索動作
    recent, losses = [], []
    best = -np.inf
    t0 = time.time()
    step, next_eval = 0, cfg.eval_every
    next_save = cfg.save_every if cfg.save_every > 0 else None
    updates_per_iter = max(1, cfg.n_envs // cfg.train_every)

    if cfg.freeze_play:
        print("出牌頭凍結：只訓練下注頭，出牌一律貪婪", flush=True)
    while step < cfg.total_steps:
        eps = 0.0 if cfg.freeze_play else linear_eps(step, cfg)
        if rollout:                          # 在目前的決策點，對所有合法動作做 rollout
            for i, env in enumerate(envs):
                if rng.random() < cfg.rollout_prob:
                    buf.push(obs[i], *agent.rollout_targets(env))
        acts, expl = agent.act_batch(obs, masks, eps, rng, return_explore=True)
        bet_x, bet_y = [], []
        for i, env in enumerate(envs):
            is_bet = env.phase == "bet"
            if regress and is_bet:
                if expl[i]:                          # 下注不需探索：改回貪婪選擇
                    acts[i] = int(agent.q_values(obs[i][None], masks[i][None])[0].argmax())
                bet_obs[i], explored[i] = obs[i].copy(), False
            elif expl[i]:
                explored[i] = True
            o2, r, done, info = env.step(acts[i])
            m2 = env.legal_mask()
            if not rollout and not cfg.freeze_play:
                r_train = info.get("expected_reward", r) if done else r
                if agent.bet_mode in ("value", "regress"):
                    r_train /= env.base_bet          # 換算成每單位下注
                if not (regress and is_bet):         # regress：下注轉移不進出牌的 buffer
                    pending[i].append((obs[i].copy(), acts[i], r_train, o2, float(done), m2))
            if done:
                if not rollout and not cfg.freeze_play:
                    G = sum(t[2] for t in pending[i])     # gamma = 1 的 Monte Carlo 回報
                    for t in pending[i]:
                        buf.push(*t, g=G)
                    pending[i] = []
                if regress and not explored[i]:      # 只用整局貪婪出牌的局訓練下注頭
                    bet_x.append(bet_obs[i])
                    bet_y.append(info.get("expected_reward", r) / env.base_bet)
                recent.append(r)
                o2, info = env.reset()
                recent.extend(info["auto_rewards"])
                m2 = env.legal_mask()
            obs[i], masks[i] = o2, m2
        step += cfg.n_envs
        if regress:
            agent.push_bet(np.array(bet_x, np.float32).reshape(-1, agent.obs_dim),
                           np.array(bet_y, np.float64))
            agent.update_bet(rng, n_steps=1)

        if buf.n >= cfg.learning_starts and not cfg.freeze_play:
            for _ in range(updates_per_iter):
                batch = buf.sample(cfg.batch_size, rng)
                losses.append(agent.update_rollout(batch) if rollout else agent.update(batch))

        if next_save is not None and step >= next_save:
            agent.save(os.path.join(cfg.out_dir, f"step_{next_save}.pt"))
            next_save += cfg.save_every

        if step >= next_eval or step >= cfg.total_steps:
            next_eval += cfg.eval_every
            res = evaluate_policy(agent, rules, obs_cfg, cfg.eval_rounds, seed=cfg.seed + 999)
            tr = float(np.mean(recent[-50_000:])) if recent else float("nan")
            ls = float(np.mean(losses[-2_000:])) if losses else float("nan")
            log.writerow([step, eps, tr, ls, res["mean_reward_per_round"], *res["ci95"],
                          res["ev_per_unit_bet"], res["mean_bet"]])
            log_f.flush()
            print(f"\nstep {step:,}  eps {eps:.3f}  訓練近期平均 {tr:+.4f}  loss {ls:.5f}  "
                  f"({time.time() - t0:.0f}s)")
            print(format_result(res), flush=True)
            agent.save(os.path.join(cfg.out_dir, "last.pt"))
            if res["mean_reward_per_round"] > best:
                best = res["mean_reward_per_round"]
                agent.save(os.path.join(cfg.out_dir, "best.pt"))
    log_f.close()
    agent.save(os.path.join(cfg.out_dir, "last.pt"))   # 確保 --steps 0（只預訓練）時也有存檔
    return agent


def main():
    p = argparse.ArgumentParser(description="訓練 Blackjack DQN")
    add_rule_args(p)
    d = TrainConfig()
    g = p.add_argument_group("訓練參數")
    g.add_argument("--steps", type=int, default=d.total_steps)
    g.add_argument("--n-envs", type=int, default=d.n_envs)
    g.add_argument("--hidden", default=",".join(map(str, d.hidden)))
    g.add_argument("--lr", type=float, default=d.lr)
    g.add_argument("--lr-end", type=float, default=None, help="學習率線性衰減的終值（預設不衰減）")
    g.add_argument("--target", default=d.target_mode, choices=["td", "mc", "rollout"],
                   help="td=Double DQN；mc=Monte Carlo 回報；rollout=每個決策點平行試所有動作")
    g.add_argument("--rollout-prob", type=float, default=d.rollout_prob,
                   help="rollout 模式下每個決策點做 rollout 的機率")
    g.add_argument("--no-expected", action="store_true", help="訓練目標改用實際結算（不用期望結算）")
    g.add_argument("--no-dueling", action="store_true")
    g.add_argument("--loss", default=d.loss, choices=["mse", "huber"],
                   help="mse：估計期望值（預設）；huber：舊版，會低估報酬偏斜的動作")
    g.add_argument("--bet-head", default=d.bet_head, choices=["regress", "value", "q"],
                   help="下注頭：regress=獨立下注頭、以實際報酬為目標（預設）；"
                        "value=共用主幹、以出牌頭 TD 為目標；q=每個倍數各一個 Q 值")
    g.add_argument("--play-prior", default=d.play_prior, choices=["none", "residual"],
                   help="residual：Q = 網路 + basic strategy 先驗，起點即 basic strategy")
    g.add_argument("--prior-margin", type=float, default=d.prior_margin,
                   help="先驗給 basic 動作的分數；網路需學到超過此值的優勢才會偏離")
    g.add_argument("--deviate-threshold", type=float, default=d.deviate_threshold,
                   help="決策門檻：Q 需比 basic 動作高出此值才偏離（只影響選擇，不影響訓練）")
    g.add_argument("--pretrain-rounds", type=int, default=d.pretrain_rounds,
                   help="RL 前用 basic strategy 打多少局做模仿學習（0 = 不預訓練）")
    g.add_argument("--pretrain-updates", type=int, default=d.pretrain_updates)
    g.add_argument("--teacher", default=d.teacher, choices=["basic", "dp"],
                   help="預訓練老師：basic=模仿動作；dp=蒸餾 DP 的各動作期望值")
    g.add_argument("--init-play", default=d.init_play,
                   help="從已訓練模型載入出牌頭（例如蒸餾模型），用來組合完整系統")
    g.add_argument("--freeze-play", action="store_true",
                   help="凍結出牌頭，只訓練下注頭；出牌一律貪婪")
    g.add_argument("--teacher-cache", default=d.teacher_cache,
                   help="老師示範資料快取檔（.npz），相同規則與局數時直接讀取")
    g.add_argument("--anchor-weight", type=float, default=d.anchor_weight,
                   help="RL 階段保留模仿學習損失的權重（0 = 不啟用），建議試 0.1、0.5")
    g.add_argument("--anchor-margin", type=float, default=d.anchor_margin)
    g.add_argument("--eps-start", type=float, default=d.eps_start,
                   help="從 basic strategy 起步時建議調低，例如 0.1")
    g.add_argument("--bet-hidden", type=int, default=d.bet_hidden, help="下注頭殘差 MLP 寬度，0=只用線性")
    g.add_argument("--bet-memory", type=int, default=d.bet_memory, help="下注頭線性部分的有效記憶局數")
    g.add_argument("--batch-size", type=int, default=d.batch_size)
    g.add_argument("--buffer", type=int, default=d.buffer_size)
    g.add_argument("--learning-starts", type=int, default=d.learning_starts,
                   help="buffer 累積多少筆後才開始訓練")
    g.add_argument("--train-every", type=int, default=d.train_every,
                   help="每收集幾步做一次梯度更新；CPU 較慢時可調大")
    g.add_argument("--eps-decay", type=int, default=d.eps_decay_steps)
    g.add_argument("--eval-every", type=int, default=d.eval_every)
    g.add_argument("--save-every", type=int, default=d.save_every,
                   help="每隔多少步另存 step_<步數>.pt，供事後比較不同訓練階段（0 = 不另存）")
    g.add_argument("--eval-rounds", type=int, default=d.eval_rounds)
    g.add_argument("--seed", type=int, default=d.seed)
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--out", default=d.out_dir)
    a = p.parse_args()

    cfg = TrainConfig(total_steps=a.steps, n_envs=a.n_envs,
                      hidden=tuple(int(x) for x in a.hidden.split(",")), lr=a.lr,
                      lr_end=a.lr if a.lr_end is None else a.lr_end, target_mode=a.target,
                      rollout_prob=a.rollout_prob, expected_settlement=not a.no_expected,
                      dueling=not a.no_dueling, bet_head=a.bet_head, loss=a.loss,
                      bet_hidden=a.bet_hidden, bet_memory=a.bet_memory,
                      play_prior=a.play_prior, prior_margin=a.prior_margin,
                      deviate_threshold=a.deviate_threshold,
                      pretrain_rounds=a.pretrain_rounds, pretrain_updates=a.pretrain_updates,
                      anchor_weight=a.anchor_weight, anchor_margin=a.anchor_margin,
                      teacher=a.teacher, teacher_cache=a.teacher_cache,
                      init_play=a.init_play, freeze_play=a.freeze_play,
                      eps_start=a.eps_start,
                      batch_size=a.batch_size, buffer_size=a.buffer, train_every=a.train_every,
                      learning_starts=a.learning_starts, eps_decay_steps=a.eps_decay,
                      eval_every=a.eval_every, eval_rounds=a.eval_rounds, save_every=a.save_every, seed=a.seed,
                      device=a.device, out_dir=a.out)
    rules, obs_cfg = rules_from_args(a), obs_from_args(a)
    print("規則:", rules)
    print("觀測:", obs_cfg)
    agent = train(rules, obs_cfg, cfg)

    from .analysis import bet_by_true_count, strategy_table
    strategy_table(agent)
    bet_by_true_count(agent)


if __name__ == "__main__":
    main()
