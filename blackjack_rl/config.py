"""規則、觀測與訓練參數。所有可調整的設定都集中在這裡。"""
from dataclasses import dataclass, field
from typing import Tuple, Union

# 觀測正規化用的最大副數。只用來縮放數值，不會把實際副數放進 state。
MAX_DECKS = 8

# 動作編號（沿用原專案與手稿：0 stick, 1 hit, 2 surrender, 3 split, 4 double）
STICK, HIT, SURRENDER, SPLIT, DOUBLE = 0, 1, 2, 3, 4
N_PLAY_ACTIONS = 5


@dataclass
class RuleConfig:
    # 副數：int 表示固定；(lo, hi) 表示每次洗牌在 [lo, hi] 之間隨機選一個整數
    n_decks: Union[int, Tuple[int, int]] = 6
    # 發到牌堆的多少比例時放 cut card（每次洗牌在區間內隨機）
    penetration: Tuple[float, float] = (0.65, 0.80)
    # cut card 之後至少保留的張數，避免單副牌時一局中途把牌發完
    min_reserve: int = 26
    burn_cards: int = 1                  # 洗牌後燒牌張數（永不對玩家公開）
    dealer_peek: bool = True             # True=美式（一明一暗 + peek）；False=歐式 ENHC
    dealer_hits_soft17: bool = False     # False=S17, True=H17
    blackjack_payout: float = 1.5        # 3:2 = 1.5；6:5 = 1.2
    double_rule: str = "any"             # "any" | "9-11" | "10-11"
    double_after_split: bool = True      # DAS
    max_hands: int = 4                   # 分牌後最多幾手
    resplit_aces: bool = False
    hit_split_aces: bool = False         # 分 A 後能否繼續抽牌
    late_surrender: bool = True
    # 下注決策：每局開始前選擇下注倍數
    enable_betting: bool = True
    bet_sizes: Tuple[float, ...] = (1, 2, 4, 8)

    def __post_init__(self):
        assert self.double_rule in ("any", "9-11", "10-11")
        lo, hi = self.deck_range
        assert 1 <= lo <= hi <= MAX_DECKS, f"n_decks 必須介於 1~{MAX_DECKS}"
        assert 0.0 < self.penetration[0] <= self.penetration[1] < 1.0
        assert 1 <= self.max_hands
        assert len(self.bet_sizes) >= 1 and all(b > 0 for b in self.bet_sizes)

    @property
    def deck_range(self) -> Tuple[int, int]:
        if isinstance(self.n_decks, int):
            return self.n_decks, self.n_decks
        return int(self.n_decks[0]), int(self.n_decks[1])

    @property
    def n_bet_actions(self) -> int:
        return len(self.bet_sizes) if self.enable_betting else 0

    @property
    def n_actions(self) -> int:
        return N_PLAY_ACTIONS + self.n_bet_actions


@dataclass
class ObsConfig:
    """state 可包含的牌堆資訊。

    無論哪一種模式，state 都「不包含」副數、剩餘總張數、各點數剩餘張數。
      - "none"        : 沒有任何牌堆資訊
      - "seen_counts" : 本次洗牌以來，玩家看得到的各點數已出現張數（10 維）
      - "hilo"        : Hi-Lo running count（1 維；不做 true count，因為需要剩餘副數）
    """
    shoe_info: str = "seen_counts"
    # 手牌點數的編碼："onehot"（4~21 點各一維）或 "scalar"（點數/21，舊版做法）
    total_encoding: str = "onehot"

    def __post_init__(self):
        assert self.shoe_info in ("none", "seen_counts", "hilo")
        assert self.total_encoding in ("onehot", "scalar")


@dataclass
class TrainConfig:
    total_steps: int = 2_000_000       # 環境互動步數
    n_envs: int = 64                   # 同時跑的環境數（批次推論加速）
    hidden: Tuple[int, ...] = (256, 256)
    lr: float = 1e-4
    lr_end: float = 1e-4               # 線性衰減到此值；等於 lr 表示不衰減
    target_mode: str = "td"            # "td"（Double DQN）、"mc"（Monte Carlo）、"rollout"（平行試所有動作）
    rollout_prob: float = 1.0          # rollout 模式下，每個決策點做 rollout 的機率
    expected_settlement: bool = True   # 訓練目標使用莊家補牌的期望結果（評估仍用實際報酬）
    dueling: bool = True
    # 下注頭（開啟下注時）：
    #   "regress"：獨立下注頭，以每局實際報酬為目標（線性精確解 + MLP 殘差），見 agent.py
    #   "value"：輸出一個數值 V(s)＝這局每單位下注的期望報酬，下 b 倍的 Q 值 = b × V(s)。
    #            出牌與下注都以「每單位下注」學習，V 以出牌頭的價值做 TD 目標。
    #   "q"    ：舊做法，每個下注倍數各自一個 Q 值
    # 出牌以 basic strategy 為起點：
    #   play_prior="residual"：Q = 網路 + 先驗（basic 的動作 + prior_margin），網路出牌頭初始為 0
    #   pretrain_rounds > 0  ：RL 前先用 basic strategy 打這麼多局做模仿學習（DQfD 式損失）
    play_prior: str = "none"
    # 決策門檻：選動作時，只有當某動作的 Q 比 basic strategy 的動作高出此值才偏離。
    # 只作用於選擇、不進入訓練目標，所以網路無法抵銷它（與 prior_margin 不同）。
    deviate_threshold: float = 0.0
    prior_margin: float = 0.02
    pretrain_rounds: int = 0
    # 預訓練的老師："basic"（模仿動作）或 "dp"（蒸餾：各動作 Q 回歸到 DP 期望值 + 間距損失）
    teacher: str = "basic"
    teacher_cache: str = ""
    # 組合完整系統：從已訓練模型載入出牌頭（init_play），並可凍結出牌頭只訓練下注頭（freeze_play）。
    # 凍結時出牌一律貪婪（不探索），下注頭學到的就是該出牌策略的實際期望值。
    init_play: str = ""
    freeze_play: bool = False            # 老師示範資料的快取檔（.npz），可跨實驗共用
    pretrain_updates: int = 5000
    pretrain_margin: float = 0.05
    # RL 階段保留模仿學習（DQfD 間距）損失：總損失 = TD 損失 + anchor_weight × 間距損失。
    # 間距損失要求 basic 動作的 Q 比其他合法動作高出 anchor_margin；0 表示不啟用。
    anchor_weight: float = 0.0
    anchor_margin: float = 0.05
    bet_head: str = "regress"
    bet_hidden: int = 16               # regress 下注頭的殘差 MLP 寬度（0 = 只用線性）
    bet_lr: float = 1e-3
    bet_batch: int = 1024
    bet_memory: int = 200_000          # 遞迴最小平方的有效記憶局數（遺忘因子 = 1 − 1/此值）
    gamma: float = 1.0                 # 只有終局獎勵，不應折扣
    batch_size: int = 512
    buffer_size: int = 500_000
    learning_starts: int = 20_000
    train_every: int = 4               # 每收集幾個 step 做一次梯度更新（以單一環境步計）
    target_update: int = 2_000         # 每幾次梯度更新同步 target network
    grad_clip: float = 10.0
    # 損失函數："mse"（預設，最佳解為期望值）或 "huber"（舊版；最佳解偏向中位數，
    # 會系統性低估報酬分布偏斜的動作，例如硬牌抽牌，導致過度投降）
    loss: str = "mse"
    huber_beta: float = 1.0
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 1_000_000
    eval_every: int = 200_000
    save_every: int = 0                # 每隔多少步另存 step_<步數>.pt（0 = 不另存）
    eval_rounds: int = 20_000
    seed: int = 0
    device: str = "cpu"
    out_dir: str = "runs/default"

    def __post_init__(self):
        assert self.target_mode in ("td", "mc", "rollout")
        assert self.loss in ("mse", "huber")
        assert self.bet_head in ("regress", "value", "q")
        assert self.play_prior in ("none", "residual")
        assert self.teacher in ("basic", "dp")
