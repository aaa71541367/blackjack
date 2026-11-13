"""學到的下注（神經網路迴歸），以及和線性下注的公平比較。

下注是單步決策，所以「學下注」就是學一個函數：可見的牌堆資訊 → 這局每單位下注的期望報酬。
NeuralBetModel 用小型 MLP 學這個函數，輸入與 DQN state 的 seen_counts 特徵相同
（各點數已出現張數 / 8 副牌時的總張數），不含副數或剩餘張數。

公平比較：兩種模型用「同一批資料、同樣的目標、同一個出牌策略」訓練，
唯一的差別是函數形式（線性 vs 神經網路）。

用法（出牌策略可選 basic、dp 或只學出牌的 DQN 模型）：
  python -m blackjack_rl.bet_learn --decks 6 --play runs/play/best.pt --rounds 2000000 --out runs/play
輸出 bet_linear.npz 與 bet_net.pt。
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from .bet_model import LinearBetModel, collect_bet_data, decide_bet
from .compat import torch_load
from .config import RuleConfig
from .env import _SEEN_SCALE


def _net_input(seen) -> np.ndarray:
    return np.asarray(seen, np.float32) / _SEEN_SCALE


class NeuralBetModel:
    name = "learned_bet"

    def __init__(self, rules: RuleConfig, hidden=(64, 64), rule: str = "ev", thresholds=None):
        self.rules, self.rule, self.hidden = rules, rule, tuple(hidden)
        n = len(rules.bet_sizes)
        self.thresholds = thresholds if thresholds is not None else [0.005 * k for k in range(n - 1)]
        layers, d = [], 10
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def fit_arrays(self, seen: np.ndarray, y: np.ndarray, epochs: int = 30, lr: float = 1e-3,
                   batch: int = 4096, weight_decay: float = 1e-4, val_frac: float = 0.1,
                   patience: int = 4, seed: int = 0, log=print):
        """以 MSE 訓練，保留一部分資料做 early stopping（訊號很弱，容易過度擬合雜訊）。"""
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        idx = rng.permutation(len(y))
        n_val = max(1, int(len(y) * val_frac))
        va, tr = idx[:n_val], idx[n_val:]
        X = torch.as_tensor(_net_input(seen))
        Y = torch.as_tensor(y, dtype=torch.float32)
        # 輸出層偏差初始化為平均值，網路只需學「偏離平均」的部分
        with torch.no_grad():
            self.net[-1].bias.fill_(float(Y[tr].mean()))
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        best, best_state, bad = np.inf, None, 0
        for ep in range(epochs):
            self.net.train()
            perm = torch.as_tensor(rng.permutation(tr))
            for i in range(0, len(perm), batch):
                b = perm[i:i + batch]
                loss = ((self.net(X[b]).squeeze(1) - Y[b]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = float(((self.net(X[va]).squeeze(1) - Y[va]) ** 2).mean())
            if v < best - 1e-7:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        self.net.load_state_dict(best_state)
        self.val_mse = best
        return self

    @torch.no_grad()
    def predict(self, seen) -> float:
        return float(self.net(torch.as_tensor(_net_input(seen))[None])[0, 0])

    def bet_index(self, seen) -> int:
        return decide_bet(self.predict(seen), self.rules, self.rule, self.thresholds)

    def save(self, path):
        torch.save({"state_dict": self.net.state_dict(), "hidden": self.hidden,
                    "rule": self.rule, "thresholds": self.thresholds}, path)

    @classmethod
    def load(cls, path, rules: RuleConfig):
        d = torch_load(path, map_location="cpu")
        m = cls(rules, d["hidden"], d["rule"], d["thresholds"])
        m.net.load_state_dict(d["state_dict"])
        m.net.eval()
        return m


def load_play_policy(spec: str, rules: RuleConfig):
    """'basic'、'dp'、'dp-exact' 或 DQN 模型路徑（.pt）。"""
    from .optimal import OptimalPlay
    from .strategies import BasicStrategy
    if spec == "basic":
        return BasicStrategy()
    if spec in ("dp", "dp-exact"):
        return OptimalPlay(rules, exact=spec == "dp-exact")
    from .agent import DQNAgent
    return DQNAgent.load(spec)


def main():
    from .cli import add_rule_args, rules_from_args

    p = argparse.ArgumentParser(description="以同一批資料訓練線性與神經網路下注模型")
    add_rule_args(p)
    p.add_argument("--play", default="basic", help="出牌策略：basic、dp、dp-exact 或 DQN 模型路徑")
    p.add_argument("--rounds", type=int, default=2_000_000)
    p.add_argument("--rule", default="ev", choices=["ev", "spread"])
    p.add_argument("--hidden", default="64,64")
    p.add_argument("--no-expected", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=".")
    a = p.parse_args()
    rules = rules_from_args(a)
    os.makedirs(a.out, exist_ok=True)
    play = load_play_policy(a.play, rules)

    print(f"收集 {a.rounds:,} 局資料（出牌策略：{a.play}）")
    seen, y = collect_bet_data(rules, play, a.rounds, a.seed, not a.no_expected)
    # 最後 10% 保留做兩種模型的共同驗證集
    n_test = len(y) // 10
    lin = LinearBetModel(rules, rule=a.rule).fit_arrays(seen[:-n_test], y[:-n_test])
    net = NeuralBetModel(rules, tuple(int(x) for x in a.hidden.split(",")), a.rule)
    net.fit_arrays(seen[:-n_test], y[:-n_test], seed=a.seed)
    lin.save(os.path.join(a.out, "bet_linear.npz"))
    net.save(os.path.join(a.out, "bet_net.pt"))

    st, yt = seen[-n_test:], y[-n_test:]
    base = float(((yt - yt.mean()) ** 2).mean())
    for name, m in [("線性", lin), ("神經網路", net)]:
        pred = np.array([m.predict(s) for s in st])
        mse = float(((yt - pred) ** 2).mean())
        bet_max = np.mean([m.bet_index(s) == len(rules.bet_sizes) - 1 for s in st])
        print(f"{name:<6} 驗證集 MSE 改善 {(base - mse) / base * 100:+.4f}%   "
              f"預測範圍 [{pred.min() * 100:+.2f}%, {pred.max() * 100:+.2f}%]   下最大注比例 {bet_max:.1%}")
    print("MSE 改善很小是正常的：單局結果的雜訊遠大於牌堆組成帶來的差異。")
    print("兩者實際下注效果請用 compare --matrix 比較。")


if __name__ == "__main__":
    main()
