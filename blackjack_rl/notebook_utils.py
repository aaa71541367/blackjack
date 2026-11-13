"""Jupyter notebook 用的輔助函式（experiment_pipeline.ipynb）。

- run_commands：以子程序（可平行）執行訓練指令，失敗時列出紀錄檔
- cached：長時間計算的結果存成 JSON，重跑時直接讀取
- paired_many / full_system_many：以多個行程平行做配對評估與完整系統評估
- pool：合併多個測試 seed 的結果（等局數的平均與合併信賴區間）
"""
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np


# ---------------------------------------------------------------- 執行訓練指令
def run_commands(cmds, logs, parallel=True, echo=True):
    """cmds：指令參數串列的串列；logs：對應的紀錄檔路徑。"""
    t0 = time.time()
    procs = []
    for cmd, log in zip(cmds, logs):
        os.makedirs(os.path.dirname(log) or ".", exist_ok=True)
        f = open(log, "w")
        full = [sys.executable, "-m"] + [str(c) for c in cmd]
        p = subprocess.Popen(full, stdout=f, stderr=subprocess.STDOUT)
        procs.append((p, f, log))
        if not parallel:
            p.wait()
    failed = []
    for p, f, log in procs:
        if p.wait() != 0:
            failed.append(log)
        f.close()
    if failed:
        raise RuntimeError("以下指令失敗，請查看紀錄檔：\n  " + "\n  ".join(failed))
    if echo:
        print(f"完成 {len(cmds)} 個指令（{(time.time() - t0) / 60:.1f} 分鐘）")


def tail(path, n=8, grep=None):
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    if grep:
        lines = [l for l in lines if grep in l]
    print("\n".join(lines[-n:]))


# ---------------------------------------------------------------- 結果快取
def cached(path, fn, refresh=False):
    """若 path 已存在就讀取，否則執行 fn() 並存成 JSON。"""
    if os.path.exists(path) and not refresh:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    t0 = time.time()
    res = fn()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"已計算並存檔：{path}（{(time.time() - t0) / 60:.1f} 分鐘）")
    return res


def _clean(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (tuple, list)):
            out[k] = [float(x) for x in v]
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _pool_ctx():
    return mp.get_context("spawn")      # 避免 fork 與 PyTorch 執行緒的衝突


# ---------------------------------------------------------------- 配對評估
def _paired_job(job):
    import torch
    torch.set_num_threads(1)
    from .analysis import load_any
    from .config import ObsConfig, RuleConfig
    from .optimal import OptimalPlay
    from .paired import paired_eval
    if job["model"] == "dp":
        rules = RuleConfig(**job["rules"])
        obs = ObsConfig(job.get("shoe_info", "seen_counts"))
        pol = OptimalPlay(rules)
    else:
        pol = load_any(job["model"])
        rules, obs = pol.rules, pol.obs_cfg
    base = OptimalPlay(rules) if job.get("baseline") == "dp" else None
    t0 = time.time()
    r = paired_eval(pol, rules, obs, job["rounds"], job["seed"], baseline=base)
    r = _clean(r)
    r.update(model=job["model"], seed=job["seed"], baseline=job.get("baseline", "basic"),
             minutes=(time.time() - t0) / 60)
    return r


def paired_many(jobs, workers=3):
    """jobs：[{model, seed, rounds, baseline?, rules?(DP 用)}]，回傳結果串列（順序同 jobs）。"""
    with ProcessPoolExecutor(max_workers=workers, mp_context=_pool_ctx()) as ex:
        return list(ex.map(_paired_job, jobs))


def pool(results):
    """合併多個等局數的配對結果：平均的平均，標準誤由各次標準誤合成。"""
    means = np.array([r["mean_diff"] for r in results])
    ses = np.array([(r["ci95"][1] - r["ci95"][0]) / 3.92 for r in results])
    m = means.mean()
    se = np.sqrt((ses ** 2).sum()) / len(results)
    return {"mean": float(m), "ci95": [float(m - 1.96 * se), float(m + 1.96 * se)], "n": len(results)}


def fmt_pct(x, ci=None):
    s = f"{x * 100:+.3f}%"
    if ci is not None:
        s += f" [{ci[0] * 100:+.3f}%, {ci[1] * 100:+.3f}%]"
    return s


# ---------------------------------------------------------------- 完整系統評估
def _full_job(job):
    import torch
    torch.set_num_threads(1)
    from .agent import DQNAgent
    from .bet_model import BaselinePolicy, DQNBetHead, LinearBetModel
    from .config import RuleConfig
    from .evaluate import evaluate_policy
    from .strategies import BasicStrategy
    ag = DQNAgent.load(job["model"])
    rules = ag.rules
    lin = LinearBetModel.load(job["bet_linear"], rules) if job.get("bet_linear") else None
    plays = {"basic": BasicStrategy(), "dqn": ag}
    bets = {"min": None, "head": DQNBetHead(ag)}
    if lin is not None:
        bets["linear"] = lin
    out = {}
    for pn, play in plays.items():
        for bn, bm in bets.items():
            r = evaluate_policy(BaselinePolicy(play, bm), rules, ag.obs_cfg, job["rounds"], job["seed"])
            out[f"{pn}+{bn}"] = {"ev_per_unit": float(r["ev_per_unit_bet"]),
                                 "round_mean": float(r["mean_reward_per_round"]),
                                 "round_ci95": [float(x) for x in r["ci95"]],
                                 "mean_bet": float(r["mean_bet"])}
    return {"model": job["model"], "seed": job["seed"], "cells": out}


def full_system_many(jobs, workers=3):
    with ProcessPoolExecutor(max_workers=workers, mp_context=_pool_ctx()) as ex:
        return list(ex.map(_full_job, jobs))
