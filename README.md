# Blackjack DQN（重構版）

以 [leonvol/blackjack-dqn](https://github.com/leonvol/blackjack-dqn) 為起點重寫。原專案的環境有多個會嚴重影響結果的 bug（見文末），因此環境、網路與訓練流程全部重寫，改用 PyTorch。

## 安裝與執行

```bash
pip install -r requirements.txt    # 支援 Python 3.7 以上、PyTorch 1.2 以上
python -m pytest tests                          # 單元測試（90 項）

# 1. 先驗證環境：basic strategy 在 6 副 S17 DAS LS 下應約為 -0.4%
python -m blackjack_rl.evaluate --policy basic --decks 6 --rounds 2000000

# 2. 訓練（預設：4~8 副隨機、state 含各點數已出現張數、含下注決策）
python -m blackjack_rl.train --decks 4-8 --shoe-info seen_counts --steps 5000000 --out runs/seen

# 3. 分析：策略表（與 basic strategy 比較）＋ 依 true count 分組的下注行為
python -m blackjack_rl.analysis runs/seen/best.pt

# 4. 評估已訓練模型
python -m blackjack_rl.evaluate --policy dqn --model runs/seen/best.pt --rounds 500000

# 5. Tabular 對照組（none 或 hilo 模式）
python -m blackjack_rl.tabular --decks 6 --no-betting --method q --rounds 20000000 --out runs/tab_q

# 6. 用相同 seed 比較多個模型（.pt 為 DQN、.pkl 為 tabular；basic strategy 自動列入）
python -m blackjack_rl.compare runs/tab_q/tabular.pkl runs/td/best.pt runs/mc/best.pt --rounds 1000000

# 7. 出牌與下注分開比較（建議流程）
#   (a) DQN 只學出牌
python -m blackjack_rl.train --decks 4-8 --no-betting --shoe-info seen_counts --steps 10000000 --out runs/play
#   (b) 用 DQN 的出牌收集資料，同一批資料訓練線性與神經網路下注
python -m blackjack_rl.bet_learn --decks 4-8 --play runs/play/best.pt --rounds 5000000 --out runs/play
#   (c) 出牌 × 下注比較矩陣
python -m blackjack_rl.compare --decks 4-8 --dp --play-model runs/play/best.pt \
    --bet-linear runs/play/bet_linear.npz --bet-net runs/play/bet_net.pt --rounds 1000000
```

有 GPU 時會自動使用 `cuda`。訓練過程記錄在 `runs/<名稱>/log.csv`，設定存在 `config.json`。

## 一鍵執行完整實驗

`run_pipeline.sh` 依序執行整個研究流程，每個階段可單獨執行，結果存在 `runs/`，文字紀錄存在 `logs/`：

```bash
bash run_pipeline.sh 0          # 環境檢查
bash run_pipeline.sh 1 2 3      # DP 上限參照、DP 蒸餾、開發 seed 檢查
bash run_pipeline.sh 4          # 出牌最終驗證
bash run_pipeline.sh 5 6 7      # 組合完整系統、線性下注對照、完整系統最終評估
bash run_pipeline.sh ablation   # 消融實驗（選用）
```

腳本開頭集中了所有設定（副數、seed、局數、是否平行）。任何一個 seed 失敗時腳本會停止並指出紀錄檔。

同樣的流程也有 Jupyter notebook 版本 `experiment_pipeline.ipynb`：評估直接呼叫 Python API 並自動合併多個測試 seed、結果快取成 `results/*.json`（重跑不會重算）、附圖表；第 9 階段為資金管理實驗（錄製序列、DP 最佳解、資金管理 DQN、成功機率比較與下注熱圖）。第一次使用建議先把 `QUICK` 設為 `True` 跑一遍，確認環境正常。

## 專案結構

| 檔案 | 內容 |
|---|---|
| `blackjack_rl/config.py` | 規則（`RuleConfig`）、觀測（`ObsConfig`）、訓練（`TrainConfig`）參數 |
| `blackjack_rl/shoe.py` | 有限牌堆：副數、cut card、燒牌、玩家可見牌的計數 |
| `blackjack_rl/env.py` | 環境：下注、發牌、peek、分牌、double、投降、莊家補牌、結算、state 編碼 |
| `blackjack_rl/strategies.py` | Basic strategy 與隨機策略 |
| `blackjack_rl/agent.py` | DQN：TD／MC／rollout 目標、dueling、學習率衰減 |
| `blackjack_rl/dealer.py` | 莊家最終結果的精確機率分布（期望結算用） |
| `blackjack_rl/tabular.py` | Tabular 對照組：Monte Carlo control、Q-learning |
| `blackjack_rl/compare.py` | 以相同 seed 比較多個策略（可加入上限基準） |
| `blackjack_rl/optimal.py` | 上限基準：只用可見資訊的 DP 出牌 |
| `blackjack_rl/bet_model.py` | 上限基準：線性迴歸下注；下注資料收集；出牌＋下注組合策略 |
| `blackjack_rl/prior.py` | basic strategy 先驗（由 state 查表，供殘差 Q 值與模仿學習使用） |
| `blackjack_rl/bankroll.py` | 資金管理（期限內翻倍）：序列錄製、DP 最佳解、資金管理 DQN、評估與熱圖 |
| `blackjack_rl/teacher.py` | 老師示範資料（basic 或 DP）收集與快取 |
| `blackjack_rl/paired.py` | 配對評估：同一牌序上與 basic strategy 比較出牌 |
| `blackjack_rl/bet_learn.py` | 學到的下注（神經網路迴歸），與線性下注用同一批資料訓練 |
| `blackjack_rl/train.py` | 訓練（多環境批次推論） |
| `blackjack_rl/evaluate.py` | 評估：平均報酬、95% 信賴區間、下注與動作分布 |
| `blackjack_rl/analysis.py` | 策略表、下注 vs. true count 分析 |
| `tests/` | 單元測試 |

## 遊戲規則設定

| 參數 | 命令列 | 預設 | 說明 |
|---|---|---|---|
| 副數 | `--decks` | 6 | `6` 固定；`4-8` 每次洗牌隨機 |
| cut card | `--penetration` | 0.65-0.80 | 發到此比例後，下一局開始前重洗 |
| 燒牌 | `--burn` | 1 | 永不對玩家公開 |
| 莊家規則 | `--enhc` | 美式 | 美式＝一明一暗＋peek；歐式＝暗牌在玩家結束後才發 |
| 軟 17 | `--h17` | S17 | |
| BJ 賠率 | `--bj-payout` | 1.5 | 6:5 用 1.2 |
| Double 條件 | `--double-rule` | any | `any` / `9-11` / `10-11` |
| 分牌後 double | `--no-das` | 允許 | |
| 最多手數 | `--max-hands` | 4 | |
| 再分 A / 分 A 後抽牌 | `--rsa` / `--hsa` | 不允許 | |
| 投降 | `--no-surrender` | late surrender | 只能在第一個決策、未分牌時 |
| 下注決策 | `--no-betting`、`--bet-sizes` | 開啟，1,2,4,8 | |

## 回合流程與獎勵

一局是一個 episode，**牌堆跨局延續**：

1. **下注**（動作 5 起，對應 `bet_sizes`）
2. **發牌**：玩家、莊家明牌、玩家、莊家暗牌（美式）。莊家 peek 到 BJ 或玩家 BJ 時直接結算
3. **玩家決策**：0 停牌、1 抽牌、2 投降、3 分牌、4 加倍。分牌後依序打每一手，第二手的第二張牌輪到它時才發
4. **莊家**：翻開暗牌並依規則補牌；所有手都爆牌或投降時不補牌
5. **結算**（`done=True`）：輸 -bet、平手 0、贏 +bet、BJ +1.5×bet、投降 -0.5×bet；分牌的各手加總

中間步驟 reward 皆為 0。點數達 21 時自動停牌。歐式規則下莊家若成為 BJ，加倍與分牌的注全輸，投降也輸全額。

## State 設計與資訊限制

State 分兩部分：

**牌局特徵（20 維）**：是否為下注階段、點數/21、是否軟牌、是否對子、是否為這手第一個決策、是否分牌後的手、手數、可否 double/split/surrender、莊家明牌 one-hot。

**牌堆特徵**（`--shoe-info`）：

| 模式 | 維度 | 內容 |
|---|---|---|
| `none` | 0 | 無 |
| `seen_counts` | 10 | 本次洗牌以來，玩家看得到的各點數已出現張數 |
| `hilo` | 1 | Hi-Lo running count |

**任何模式下，state 都不包含副數、剩餘總張數、各點數剩餘張數。** 可見牌的規則是：自己的牌、莊家明牌、分牌的其他手在發出時計入；莊家暗牌翻開時才計入；燒牌永不計入。`hilo` 模式只給 running count，不給 true count，因為 true count 需要除以剩餘副數。

`tests/test_env.py::test_obs_independent_of_deck_count` 驗證：看到同樣的牌時，4 副與 8 副牌的 state 完全相同。

**注意**：若訓練時副數固定，網路雖看不到副數，仍可能從「已出現張數」隱性學出剩餘張數。要確實做到不知道剩多少，請用 `--decks 4-8` 讓副數隨機。

## DQN 目標值與 tabular 對照組

**DQN 目標值**（`--target`）：

- `td`（預設）：Double DQN，y = r + Q_target(s′, argmax Q(s′, ·))
- `mc`：Monte Carlo，y = 該局總報酬。每局結束後才把該局所有 transition 連同回報放入 buffer

`--lr-end` 讓學習率在訓練過程中線性衰減，例如 `--lr 1e-4 --lr-end 1e-5`。

**Tabular 對照組**（`--method`）：`mc` 為 every-visit Monte Carlo control，`q` 為 Q-learning，步長皆為 1/N(s,a)^p（`--alpha-power`，預設 0.7）。state key 與 DQN 遵守同樣的資訊限制；`seen_counts` 為 10 維計數，組合數太大，tabular 不支援。

**Monte Carlo 的探索偏差**：抽牌與分牌之後還有後續決策，訓練時這些決策有 ε 機率隨機，會拉低回報；停牌、投降、加倍則直接結束不受影響。所以 MC 會系統性低估抽牌與分牌、高估投降。Q-learning 用下一狀態的最佳值當目標，沒有這個偏差。

### 初步實驗（6 副 S17 DAS LS、無下注、無牌堆資訊、單顆 CPU）

以同一個 seed 評估 30 萬局，每局 95% 信賴區間約 ±0.4%：

| 策略 | 訓練量 | 每單位下注 EV | 與 basic strategy 一致率 |
|---|---|---|---|
| Basic strategy | — | -0.53% | 100% |
| Tabular Q-learning | 300 萬局 | -1.33% | 81.5% |
| Tabular MC | 300 萬局 | -1.36% | 73.9% |
| Tabular Q-learning | 43 萬局 | -2.54% | 77.0% |
| DQN（TD） | 64 萬步 ≈ 43 萬局 | -2.93% | 69.4% |
| DQN（MC） | 64 萬步 ≈ 43 萬局 | -3.07% | 68.5% |

解讀：

- 在相同資料量下，DQN 比 tabular Q-learning 略差，但差距不大；主要瓶頸是**樣本數**，連 tabular 用 300 萬局也還沒收斂
- DQN 的 TD 與 MC 目標在這個訓練量下沒有明顯差異；MC 的探索偏差在 tabular 的策略表上看得很清楚（分牌過少、投降過多）
- 建議在 GPU 上以千萬步級別訓練，並優先使用 `--target td`
- 下一節的期望結算與網路改良大幅縮短了所需的訓練量

## 降低目標雜訊與網路改良

預設全部開啟：

- **期望結算**（關閉：`--no-expected`）：局結束時，用牌堆真實組成與暗牌精確計算莊家補牌結果的機率分布，以期望報酬作為訓練目標。只用於訓練，評估一律用實際報酬；state 仍看不到剩餘牌。數學上，對隱藏資訊取條件期望再平均等於只用可見資訊的期望值，所以目標無偏、雜訊較小。`tests/test_variance_reduction.py` 驗證了期望值與重新洗牌後實際結算的平均一致
- **點數 one-hot**（`--total-encoding scalar` 改回舊版）：4~21 點各一維
- **Dueling 架構**（關閉：`--no-dueling`）：Q = V + A − mean(A)

另有實驗性的 `--target rollout`：每個決策點複製環境，對所有合法動作各跑一次（共用相同牌序）。目前的 Python 實作計算成本高，效果不如預設設定，不建議使用。

### 對照實驗（6 副、無下注、無牌堆資訊、各訓練 32 萬步、同 seed 評估 20 萬局）

| 組別 | 每單位下注 EV | 一致率 |
|---|---|---|
| Basic strategy | -0.48% | 100% |
| 舊設定（scalar、無 dueling、實際結算） | -2.79% | 73.0% |
| + one-hot + dueling | -1.90% | 79.7% |
| **+ one-hot + dueling + 期望結算（新預設）** | **-0.99%** | **85.2%** |
| rollout 目標（相近執行時間） | -3.32% | 71.8% |
| Tabular Q-learning（300 萬局） | -1.25% | 81.5% |

## 上限基準（DP 出牌 + 線性迴歸下注）

兩者都只用 DQN 能看到的資訊（自己的牌、莊家明牌、各點數已出現張數，以及規則設定的副數範圍與 cut card 範圍），不讀取真實牌堆、實際副數或暗牌。`tests/test_baselines.py` 驗證：規則先驗相同時，真實牌堆是 4 副或 8 副，DP 的期望值完全相同。

**DP 出牌**（`optimal.py`）：

1. 副數後驗：依多變量超幾何機率與「尚未發到 cut card」的條件計算
2. 每種副數下，未見牌池 = 總張數 − 已出現（暗牌、燒牌視為未見牌池中的牌）；美式規則以莊家不是 BJ 為條件
3. 計算停牌、抽牌、加倍、投降、分牌的期望值，依後驗加權後取最大

近似：分牌以「兩手獨立、不再分」估計。預設快速模式在一手牌進行中不移除抽到的牌（`--dp-exact` 使用精確模式）。驗證結果：剛洗好的 6 副與 4~8 副牌上都與 basic strategy **100% 一致**；牌堆中後段抽樣 301 個決策，快速與精確模式選的動作全部相同，期望值平均差 0.066%。

**線性迴歸下注**（`bet_model.py`）：以下注時的組成偏差預測該局期望報酬（目標用期望結算），特徵為 `[1, 已出現總數, 各點數比平均多出現的張數（以 8 點為參照）]`。預設規則是預測值為正時下最大注、否則最小注（期望值最大化、不考慮風險），也可用 `--rule spread` 分段加注。

### 初步結果（6 副、seen_counts、下注 1/2/4/8、同 seed 評估 15 萬局）

| 策略 | 每單位下注 EV | 平均下注 |
|---|---|---|
| Basic strategy（最小注） | -0.55% | 1.00 |
| DP 出牌（最小注） | -0.21% | 1.00 |
| Basic + 線性下注 | +0.55% | 3.22 |
| **DP 出牌 + 線性下注** | **+1.05%** | 3.23 |
| DQN（32 萬步） | -3.06% | 5.49 |

- 只用可見資訊就能在這組規則下取得正期望值，主要來自下注調整；DP + 線性下注的每局報酬信賴區間完全在 0 以上
- DP 出牌優於 basic strategy 的幅度符合預期，但 15 萬局不足以確認
- 下注模型用 42 萬局訓練，係數方向與文獻一致（看到越多 2~6 期望值越高，看到越多 10 與 A 越低），但數值仍有雜訊，建議用數百萬局訓練
- 短訓練的 DQN 尚未學會下注，一半以上的局下最大注，是目前與上限差距最大的部分

## 出牌與下注分開比較

下注不影響牌的流動，所以出牌和下注可以完全分開學習與評估：

- **出牌**：DQN 在關閉下注的環境訓練（`--no-betting`）。這樣的模型放進有下注的環境時，下注階段會自動下最小注、出牌階段只使用前 5 個動作
- **下注**：`bet_learn.py` 用指定的出牌策略收集資料，在**同一批資料、同樣的目標**上訓練線性與神經網路兩種下注模型，唯一差別是函數形式
- **比較矩陣**（`compare.py`）：列是出牌策略（basic／DP／DQN），欄是下注方式（最小注／線性／學到的），每格都用同一個 seed 評估。看同一欄比較出牌，看同一列比較下注

### 初步結果（6 副、seen_counts、DQN 出牌 32 萬步、下注資料 25 萬局、評估 8 萬局）

每單位下注 EV：

| 出牌 \ 下注 | 最小注 | 線性下注 | 學到的下注 |
|---|---|---|---|
| Basic strategy | -0.73% | +0.07% | -0.11% |
| DP 出牌 | -0.22% | **+0.59%** | +0.52% |
| DQN 出牌 | -1.75% | -0.08% | -0.89% |

平均下注：線性 2.07 倍、學到的 1.39 倍。

- **DQN 出牌比 DP 出牌差約 1.5%**，出牌一致率 80.9%，仍需長時間訓練
- **學到的下注在三種出牌策略下都不如線性下注**。驗證集上線性模型的 MSE 改善約為神經網路的 35 倍。神經網路在小資料下保守，預測範圍較窄，下最大注的比例只有 6%（線性為 16%）
- 8 萬局的信賴區間很寬（每局約 ±0.8%，下注後更寬），以上差異多半還不顯著，需要百萬局級別的評估確認

## 兩頭 DQN：一邊學出牌、一邊學下注

### 目前預設：獨立下注頭（`--bet-head regress`）

診斷舊版（`value`）下注頭時發現三個問題：出牌頭因最大化偏差高估價值約 2.2%，下注頭的 TD 目標直接繼承這個偏差；下注頭的目標隨起手牌大幅變動，訊號只佔很小一部分，收斂極慢（V 平均 -2.74%，實際 -1.16%）；而且共用主幹缺乏合適的歸納偏差，對牌堆的反應只有線性模型的 40%。

改進後的下注頭：

1. **目標改用每局實際的每單位報酬**（期望結算），只使用整局出牌皆為貪婪選擇的局，不再從出牌頭 bootstrap，因此不繼承出牌頭的偏差
2. **與出牌網路完全分開**，輸入是「組成偏差」特徵（與線性下注模型相同）
3. **線性部分以遞迴最小平方法線上精確求解**，含遺忘因子（`--bet-memory`，預設有效記憶 20 萬局），讓出牌較差時期的舊資料逐漸淡出；小型 MLP（`--bet-hidden`，預設 16，0 為只用線性）學殘差，輸出層初始化為 0
4. 出牌頭仍用 TD，下注轉移不進入出牌的 replay buffer

改進前後（6 副、seen_counts、各訓練 32 萬步、同 seed 評估 10 萬局）：

| 項目 | 舊版（value） | 改進後（regress） |
|---|---|---|
| 下注頭 V 平均 vs 實際 | -2.74% vs -1.16% | **-1.14% vs -1.10%** |
| V 隨 true count（−4 到 +6）的變化幅度 | 2.0% | 6.7% |
| 平均下注：true count −4 → +6 | 1.01 → 1.38 | 1.00 → 7.80 |
| 完整系統（DQN 出牌 + DQN 下注頭） | -1.83% | **-1.27%** |
| DQN 下注頭 − 線性下注（basic 出牌） | -1.12% | -0.18% |
| DQN 下注頭 − 線性下注（DQN 出牌） | -0.59% | -0.18% |

下注頭與線性下注的差距已在雜訊範圍內。完整系統與「basic strategy + 線性下注」（+0.22%）剩下的差距，主要來自出牌（DQN 出牌以最小注評估為 -2.22%，basic strategy 為 -0.77%）。

### 設計原理

開啟下注時，網路共用主幹，分出兩個頭：

- **出牌頭**：5 個動作的 Q 值（可選 dueling）
- **下注頭**：只輸出一個數值 V(s)＝這局每單位下注的期望報酬。下 b 倍的 Q 值定義為 b × V(s)，所以 V > 0 時下最大注、否則最小注

訓練時所有報酬都換算成「每單位下注」，因此：

- 出牌頭學到的價值與下注倍數無關
- 下注頭的 TD 目標就是發牌後出牌頭的最佳 Q 值，剩下的隨機性只有起手牌，雜訊遠小於用整局輸贏當目標
- 下注不影響發牌，所以**不需要探索下注**，每一局都能用來學 V

舊做法（每個下注倍數各一個 Q 值）保留為 `--bet-head q`。在比較矩陣中，兩頭模型的下注頭會自動成為一欄「DQN 下注頭」，也能和其他出牌策略組合。這個流程完全不需要 DP；DP 只是選用的比較尺（`--dp`）。

### 初步結果（6 副、seen_counts、各訓練 32 萬步、同 seed 評估 10 萬局）

| 系統 | 每單位下注 EV | 平均下注 |
|---|---|---|
| 舊做法：每個倍數各一個 Q 值 | -3.06% | 5.49 |
| **兩頭 DQN（出牌頭 + 下注頭）** | **-1.83%** | 1.03 |
| 兩頭 DQN 的出牌 + 線性下注 | -1.24% | 2.06 |
| Basic strategy + 線性下注 | +0.22% | 2.06 |

- 下注頭學到了正確方向：平均下注隨 true count 單調上升（true count −4 時 1.01 倍、+6 時 1.38 倍），但仍很保守，99.4% 的局下最小注
- 舊做法在期望值為負時大量下最大注；兩頭設計修正了這個問題，完整系統改善約 1.2%
- 下注頭目前仍不如線性下注。它學的 V 以出牌頭為基礎，出牌頭還不準（一致率 78.8%）時，V 會被系統性低估，讓下注偏保守

## 以 basic strategy 為起點

從零訓練的 DQN 出牌一直比 basic strategy 差，而出牌能改善的空間本來就只有約 0.3%。所以提供兩種從 basic strategy 出發的方式，讓訓練集中在「牌堆資訊何時值得偏離」：

- **模仿學習預訓練**（`--pretrain-rounds N`）：RL 前先用 basic strategy 打 N 局，以 DQfD 式損失預訓練出牌頭（專家動作的 Q 回歸到實際回報，並要求比其他合法動作高出 `--pretrain-margin`）；這些轉移也放入 replay buffer 作為 RL 的起始資料
- **殘差 Q 值**（`--play-prior residual`）：Q = 網路 + 先驗，先驗是 basic strategy 的動作加 `--prior-margin`，網路出牌頭初始為 0，所以起點完全等於 basic strategy

從好策略起步時建議把 `--eps-start` 調低（例如 0.1）。

### 配對評估（`paired.py`）

一般評估的雜訊（每局 ±0.7% 等級）遠大於出牌能改善的空間。配對評估在每局開始時複製環境，讓候選策略與 basic strategy 在同一牌序上各打一次，只記錄報酬差；決策相同的局差值恰為 0。搭配期望結算，8 萬局就能把信賴區間縮到約 ±0.2%。

### 結果（6 副、seen_counts、只學出牌、同 seed 配對 8 萬局）

| 出牌策略 | 相對 basic strategy | 95% CI | 決策不同的局 | 判定 |
|---|---|---|---|---|
| DP 出牌（近似上限） | **+0.32%** | [+0.18%, +0.46%] | 6.4% | 優於 basic |
| DQN：模仿學習預訓練 + RL（25.6 萬步） | -0.10% | [-0.28%, +0.09%] | 9.1% | 無法區分 |
| DQN：殘差 Q 值（32 萬步） | -0.71% | [-0.93%, -0.49%] | 12.6% | 劣於 basic |
| DQN：從零訓練（32 萬步） | -0.86% | [-1.11%, -0.61%] | 16.3% | 劣於 basic |

- **沒有任何 DQN 版本在出牌上超越 basic strategy**；模仿學習版本已追上到統計上無法區分
- DP 的 +0.32% 顯著，證實只用可見資訊確實有改善空間，也證實配對評估能偵測到這個量級的差距
- 殘差版本起點等於 basic strategy，但 `--prior-margin 0.02` 不足以抵擋 Q 值估計雜訊，學出許多有害的偏離；可嘗試加大 margin
- 各版本決策與 basic 不同時平均損失 1%~5.6%，DP 不同時平均獲益 5%：DQN 的偏離多半是雜訊，DP 的偏離才是真正的算牌偏離

### 殘差 margin 與決策門檻

**殘差 margin 無法約束偏離。** margin 0.02、0.05、0.10 訓練後，與 basic 的一致率分別為 85.2%、82.7%、86.1%，配對評估分別為 -0.71%、-0.61%、-0.51%（信賴區間大幅重疊，全部顯著劣於 basic）。診斷發現網路會學著抵銷先驗：原始輸出中 basic 動作與其他動作的差距加回 margin 後，三個模型分別是 0.553、0.564、0.559，幾乎相同。因為殘差網路的 TD 目標是讓「網路 + 先驗」等於真正的 Q 值，任何固定的先驗最終都會被吸收，margin 只影響起點。

**決策門檻**（`--deviate-threshold τ`）才是正確的做法：網路照常學 Q 值，選動作時只有當某動作的 Q 比 basic 動作高出 τ 才偏離。門檻不進入訓練目標，網路無法抵銷；而且只影響選擇，可以直接套用在已訓練的模型上，用 `paired.py --thresholds 0,0.1,0.2` 比較，不需重新訓練。

配對評估結果（同 seed 8 萬局）：

| 模型 | τ | 相對 basic | 95% CI | 偏離的局 | 偏離時平均差 |
|---|---|---|---|---|---|
| 模仿學習 | 0 | -0.10% | [-0.28%, +0.09%] | 9.1% | -1.08% |
| 模仿學習 | 0.03 | -0.01% | [-0.17%, +0.14%] | 5.8% | -0.23% |
| 模仿學習 | 0.1 | +0.02% | [-0.07%, +0.11%] | 1.8% | +1.23% |
| **模仿學習** | **0.2** | **+0.045%** | [-0.008%, +0.098%] | 0.6% | **+7.97%** |
| 殘差 margin 0.1 | 0.2 | -0.03% | [-0.10%, +0.04%] | 0.8% | -4.21% |
| DP 出牌（參照） | — | +0.32% | [+0.18%, +0.46%] | 6.4% | +4.98% |

- 門檻越高，模仿學習模型的偏離越少但品質越好；τ = 0.2 時偏離平均獲益 7.97%，與 DP 的偏離品質相當，整體接近顯著優於 basic
- 殘差模型即使加上門檻，高信心的偏離仍然虧損，代表它的 Q 值排序本身不可靠；模仿學習是較好的起點
- 目前的限制在「偏離的數量」：DP 在 6.4% 的局偏離，DQN 在可信的門檻下只有 0.6%，剩下的改善空間需要更長的訓練讓 Q 值更準，才能在較低門檻下仍保持偏離品質

## DP 蒸餾 + RL 微調

以 DP 出牌為老師（`--teacher dp`），只用可見資訊，所以蒸餾後的 DQN 仍遵守同樣的資訊限制：

1. **收集示範**：DP 出牌打 `--pretrain-rounds` 局，記錄每個決策各合法動作的 DP 期望值；可用 `--teacher-cache` 存檔共用（會檢查規則是否一致）
2. **蒸餾**：各動作的 Q 值回歸到 DP 期望值（目標幾乎無雜訊），並加上間距損失確保 DP 選的動作排最前
3. **RL 微調**：照常以 TD 訓練，`--anchor-weight` 以 DQfD 方式從示範資料抽樣計算間距損失，持續約束
4. **評估**：`paired.py --baseline dp` 直接檢驗是否超越老師

使用 DP 老師時，決策門檻 τ 應設為 0（門檻是以 basic strategy 為基準，會把 DP 式的偏離擋掉）。

容器結果（6 副、seen_counts、DP 示範 6 萬局 / 8.6 萬個決策、同 seed 配對評估）：

| 模型 | 相對 basic | 相對 DP |
|---|---|---|
| 只做 DP 蒸餾 | **+0.162%** [+0.042%, +0.281%] | -0.175% [-0.280%, -0.070%] |
| DP 蒸餾 + RL 微調（λ = 0.1、25.6 萬步） | **+0.259%** [+0.114%, +0.404%] | +0.003% [-0.142%, +0.148%] |

- 兩者都**顯著優於 basic strategy**，這是 DQN 第一次在出牌上超越 basic
- 只做蒸餾時，DQN 顯著不如 DP（約學到 DP 改善量的一半）；RL 微調後與 DP 已無法區分，相對 DP 的差距從 -0.175% 變為 +0.003%，兩者的信賴區間不重疊，所以 **RL 微調顯著改善了蒸餾模型**
- RL 微調後仍未超越 DP。DP 本身有近似（莊家機率不隨玩家抽牌更新、分牌以獨立兩手估計），理論上 RL 有超越的空間，但這需要更長的訓練與更大量的評估才能檢驗

## 基準表與模型驗證

**基準表**（`baselines.py`）：basic、DP、隨機策略在多個獨立測試 seed 上的絕對期望值（實際輸贏），加上每個 seed 的 DP 相對 basic 配對差；各 seed 合併後報告，並比較「seed 間標準差」與「單次標準誤」以確認結果不依賴特定牌序。

**下注模型驗證**（`bet_eval.py`）：
- 預測能力：以與訓練不同 seed 的獨立牌局（期望結算）做校準分析：10 個等級的預測平均 vs 實際平均、校準斜率與截距、相關係數、最高與最低等級的實際報酬差
- 實際下注效果：所有下注模型在同一條錄製序列（實際輸贏）上計算報酬，共用相同的「預測值 → 下注」規則，差異只來自預測值；信賴區間以 1,000 局為區塊做 bootstrap

```bash
python -m blackjack_rl.baselines --decks 4-8 --seeds 2027,2028,2029 --json runs/baselines.json
python -m blackjack_rl.bet_eval --decks 4-8 --play runs/full_s1/last.pt --linear runs/full_s1/bet_linear.npz \
  --net runs/full_s1/bet_net.pt --head runs/full_s1/last.pt --calib-seed 3034 --seq-seed 4045
```

## 資金管理：展示 RL 的序列決策

出牌可以用 DP 精確規劃，下注在「資金無限、期望值最大化」下是單步的完整資訊問題，兩者都不需要 RL。但資金有限、有目標與期限時，最佳下注同時取決於**牌面優勢、目前資金、剩餘局數**，而且現在的下注會改變未來的資金，這才是 RL 適合的序列決策。

**問題**：起始資金 50 單位，500 局內達到 100 單位即成功；資金低於最小注即破產；每局必須下注（1/2/4/8/16 單位，不得超過資金）。目標是最大化成功機率。（若改成最大化對數資金，最佳解是短視的 Kelly 準則，展示不出序列決策的價值。）

**分工**：出牌固定用蒸餾出牌頭，下注頭改當「牌面優勢」的預測器，資金管理 DQN 依（資金、剩餘局數、牌面優勢）決定下注。

**關鍵實作**：下注不影響發牌，所以「每局的優勢預測與每單位實際結果」可以事先錄製一次，DQN 訓練、DP 求解、所有策略評估都重複使用，而且所有策略在完全相同的牌序上配對比較。

```bash
# 錄製訓練與測試序列（不同 seed）
python -m blackjack_rl.bankroll record --decks 4-8 --play runs/full_s1/last.pt --edge runs/full_s1/last.pt \
  --rounds 3000000 --seed 11 --out runs/bank_train.npz
python -m blackjack_rl.bankroll record --decks 4-8 --play runs/full_s1/last.pt --edge runs/full_s1/last.pt \
  --rounds 1000000 --seed 22 --out runs/bank_test.npz
# DP 上限參照、訓練 DQN、評估、熱圖
python -m blackjack_rl.bankroll dp --seq runs/bank_train.npz --out runs/bank_dp.npz
python -m blackjack_rl.bankroll train --seq runs/bank_train.npz --eval-seq runs/bank_test.npz \
  --episodes 1000000 --out runs/bank_dqn
python -m blackjack_rl.bankroll evaluate --seq runs/bank_test.npz --dp runs/bank_dp.npz \
  --dqn runs/bank_dqn/model.pt --episodes 50000
python -m blackjack_rl.bankroll heatmap --dqn runs/bank_dqn/model.pt --dp runs/bank_dp.npz --out heatmap.png
```

DQN 預設使用 Double DQN（`--method td`）、dueling、每步 4 次更新，並有 50% 的訓練 episode 從隨機資金與剩餘局數開始（`--explore-starts`）。DP 把優勢分成 12 個等機率等級，由錄製序列估計轉移，是近似 MDP 的最佳解。

`evaluate` 的報告內容：
- **結局分類**：成功、破產、期限到期
- **期末資金分布**（5/25/50/75/95 分位與直方圖）與**最大回撤**（單位與比例；平均、中位數、95 分位）
- **不重疊起點**（預設）：episode 之間不共用牌局，信賴區間正確；不重疊 episode 數 = 序列局數 / 期限，所以測試序列建議 1000 萬局。`--starts random` 可取更多 episode，信賴區間改以區塊 bootstrap 計算
- **訓練 vs 測試序列**（`--train-seq`）：兩者差距反映過擬合；應以不學習的「目前規則」在兩序列上的差距作為序列難度差的對照
- **敏感度網格**（`--grid-starts`、`--grid-horizons`，目標固定）：DQN 以最長期限訓練、DP 以最長期限求解，同一組策略即可評估整個網格；DQN 的狀態以模型自己的期限正規化
- 所有策略決策時都只看可見資訊，沒有策略看得到未來的牌；「訓練序列」表現只代表在見過的牌局上評估

簡化：分牌、加倍的額外下注不受資金限制，因此期末資金可能略低於 0（視為破產）。

容器結果（6 副、basic 出牌 + 線性優勢預測、同牌序 2 萬個 episode）：

| 策略 | 成功機率 | 相對目前規則 |
|---|---|---|
| 固定最小注 | 3.7% | -42.9 pt |
| Kelly 比例下注 | 3.8% | -42.9 pt |
| 期望值為正下最大注（目前規則） | 46.6% | — |
| Bold play（下注到達標所需） | 49.4% | +2.8 pt [+2.0, +3.6] |
| DQN：Monte Carlo 目標 | 47.5% | +0.8 pt [+0.0, +1.6] |
| DQN：Double DQN + exploring starts | 48.4% | +1.8 pt [+1.0, +2.6] |
| **DQN：再加上 dueling、每步 4 次更新** | **49.5%** | **+2.8 pt [+2.0, +3.7]** |
| DP 最佳解（上限參照） | 50.8% | +4.2 pt [+3.3, +5.0] |

- DQN 顯著優於目前的下注規則，改善量與 Bold play 相當，約達 DP 改善量的三分之二
- 熱圖（`bankroll_heatmap_example.png`）顯示 DP 的策略高度依賴牌面與時間：優勢為負且時間充裕時下最小注等待，接近期限才冒險；DQN 主要學到「依資金決定下注」的粗略結構，只在低資金區開始出現依賴牌面與時間的行為，這是它與 DP 差距的來源
- Kelly 在這個目標下幾乎等於最小注，因為優勢通常只有 1% 左右

## 完整系統：蒸餾出牌頭 + 下注頭

RL 微調會損害蒸餾好的出牌頭，所以組合時將出牌頭凍結、只訓練下注頭：

```bash
python -m blackjack_rl.train --decks 4-8 --shoe-info seen_counts \
  --init-play runs/distill_s1/last.pt --freeze-play --bet-memory 100000000 \
  --steps 5000000 --eval-every 1000000 --out runs/full_s1
```

- `--init-play`：從蒸餾模型載入出牌頭（state 設定、hidden、dueling 需一致）
- `--freeze-play`：不更新出牌頭，出牌一律貪婪，下注頭學到的就是這個出牌策略的實際期望值
- 出牌策略固定後資料分布不再變動，下注頭的線性部分可用很長的記憶（`--bet-memory` 設大，接近完整最小平方）
- 凍結時沒有 TD 更新，訓練紀錄中的 loss 會顯示 nan，屬正常

容器結果（6 副、seen_counts、下注 1/2/4/8、下注頭約 48 萬局、同 seed 評估 15 萬局）：

| 出牌 \ 下注 | 最小注 | 線性下注 | DQN 下注頭 |
|---|---|---|---|
| Basic strategy | -0.552% | +0.237% | +0.567% |
| DQN（DP 蒸餾） | -0.290% | +0.555% | **+1.014%** |

- **完整系統（蒸餾出牌頭 + 下注頭）每單位下注 +1.014%，每局報酬的 95% 信賴區間 [+0.33%, +4.78%] 完全在 0 以上**，與先前的上限參照「DP 出牌 + 線性下注」（+1.05%）相當
- 平均下注：true count ≤ −2 時約 1 倍、+1 時 4.3 倍、+4 以上接近 8 倍
- 下注頭優於線性下注約 0.3%~0.46%，但兩者資料量不同（下注頭約 48 萬局、線性 25 萬局），且差距遠小於信賴區間，不能視為下注頭較好
- 下注規則為「期望值為正就下最大注」，只追求期望值、不考慮資金風險

## RL 階段保留模仿學習損失、依步數另存模型

- `--anchor-weight λ`：RL 的總損失 = TD 損失 + λ × DQfD 間距損失（要求 basic 動作的 Q 比其他合法動作高出 `--anchor-margin`，預設 0.05）。和殘差 margin 不同，這是持續作用的約束，網路無法用固定輸出抵銷；粗略地說，λ 約相當於讓 Q 值偏向 basic 動作 λ/2
- `--save-every N`：每 N 步另存 `step_<步數>.pt`，用 `paired.py` 比較不同訓練階段

容器驗證（6 副、模仿學習預訓練、MSE、25.6 萬步、同 seed 配對評估）：

| λ | 一致率 | 相對 basic（τ = 0） | 偏離的局 | 偏離時平均差 |
|---|---|---|---|---|
| 0 | 88.5% | -0.157% | 7.9% | -1.98% |
| 0.1 | 97.3% | -0.015% | 0.9% | -1.69% |
| 0.5 | 100% | -0.015% | 0.1% | 虧損（樣本極少） |

- 間距損失有效地防止了雜訊偏離：λ = 0.1 時偏離從 7.9% 降到 0.9%，整體從顯著劣於 basic 變成無法區分
- λ = 0.5 太強，幾乎完全不偏離
- 在這個訓練量下，剩下的偏離平均仍略為虧損，所以還沒有超越 basic strategy；需要更長的訓練觀察偏離品質是否轉正

## 損失函數：MSE（預設）與 Huber 的偏差

早期版本使用 Huber loss（β = 1）。Huber 的最佳解偏向報酬分布的中位數而非平均數，對「多數結果為 -1、少數為正」的偏斜分布會系統性低估。以 DP 真實值計算硬牌抽牌的 TD 目標分布，Huber 的偏差為 -0.034~-0.068，與模型實測的抽牌低估（平均 -0.05~-0.06）吻合。投降的報酬恆為 -0.5 不受影響，因此在抽牌期望值接近 -0.5 的局面（硬 14~16 對 9/10/A 等）被錯誤地學成投降；加倍與分牌的報酬尺度更大，受影響更嚴重。

現在預設使用 MSE（`--loss mse`，最佳解即期望值）；舊模型載入時會自動標記為 Huber。同設定（6 副、模仿學習、25.6 萬步）比較：

| | Huber | MSE |
|---|---|---|
| 抽牌 Q 值平均誤差 | -0.035 | -0.017 |
| 投降率（basic 為 3.4%） | 約 2.5 倍 | 2.1% |
| 配對評估 τ = 0 | -0.10% | -0.16%（信賴區間重疊） |
| 配對評估 τ = 0.1 | +0.02% | +0.02% |

MSE 修正了造成過度投降的系統性偏差，但在此訓練量下整體出牌品質沒有明顯改變；剩下的誤差主要是估計雜訊，以及取最大值造成的高估（停牌、分牌略為高估）。

## 相對原專案修正的問題

環境：
- A 的點數計算錯誤：A,A 被算成 22（爆牌）
- `step()` 先設 `new=False` 才執行動作，導致 double 完全無效、split 變成丟掉一張牌並加倍下注
- 平手被算成輸
- 沒有 Blackjack 3:2、沒有 peek
- 無限牌堆（`random.choice`），無法設定副數
- 投降在任何時候都合法；`assert` 範圍多一個
- 玩家爆牌後莊家仍補牌
- 「有無 A」改為「是否軟牌」

訓練：
- gamma 由 0.09 改為 1.0
- 加入 target network（Double DQN）
- 輸出層改為線性（原本經過 BatchNorm + LeakyReLU）
- 非法動作遮罩改為 -inf
- 評估時 reward 陣列為 int，導致投降的 -0.5 被截成 0；評估局數上限 5 步也移除
- 原 `Comparison.py` 有多處 import 與參數名稱錯誤，無法執行

## 已知限制與建議

- **期望值的差異很小，需要大量局數才能分辨。** 單局報酬標準差約 1.15，要把 95% 信賴區間縮到 ±0.1% 需要約 500 萬局。比較不同設定時，請用相同 `--seed` 並增加局數
- **下注決策的訊號非常弱。** 即使有算牌資訊，高 count 的局數比例也很低，DQN 需要很多步才能學到加注。可先用 `--no-betting --shoe-info none` 確認模型能學到 basic strategy，再逐步加入牌堆資訊與下注
- Basic strategy 基準以多副 S17 DAS LS 為準，其他規則下僅為近似
- 保險（insurance）與多名玩家同桌未實作
