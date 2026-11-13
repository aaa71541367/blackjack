#!/usr/bin/env bash
# =============================================================================
# Blackjack DQN 完整實驗流程
#
# 用法：
#   bash run_pipeline.sh 0            # 只跑第 0 階段
#   bash run_pipeline.sh 2 3 4        # 依序跑第 2、3、4 階段
#   bash run_pipeline.sh all          # 跑主要流程（0~7，不含消融實驗）
#   bash run_pipeline.sh ablation     # 消融實驗（論文中的負面結果與對照）
#   bash run_pipeline.sh 6b 9         # 下注模型驗證、資金管理
#
# 所有輸出存在 runs/，每個階段的文字結果另存在 logs/。
# 各階段可獨立重跑；已存在的 DP 示範快取會自動沿用。
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------- 共用設定
DECKS="4-8"                       # 副數隨機，確保 state 不隱含剩餘張數
SHOE="seen_counts"                # state 包含各點數已出現張數
TRAIN_SEEDS="1 2 3"               # 訓練 seed（驗證方法的穩定性）
DEV_SEED=777                      # 開發 seed：只用來檢查與選參數
TEST_SEEDS="2027 2028 2029"       # 測試 seed：只在最終驗證使用一次
DP_ROUNDS=200000                  # 每個訓練 seed 的 DP 示範局數
DISTILL_UPDATES=20000
BET_STEPS=5000000                 # 下注頭訓練步數（約等於 200 萬局）
BET_LINEAR_ROUNDS=2000000         # 線性下注對照的局數（與下注頭看過的局數相近）
FINAL_ROUNDS=1000000              # 最終評估局數
PARALLEL=1                        # 1 = 不同 seed 平行執行（需足夠 CPU 核心）

RULE="--decks $DECKS"
mkdir -p runs logs

run_seeds () {   # 對每個訓練 seed 執行函式 $1，依 PARALLEL 決定是否平行；任何一個失敗就停止
  local fn=$1 pids=() seeds=() fail=0
  for s in $TRAIN_SEEDS; do
    if [ "$PARALLEL" = "1" ]; then
      $fn "$s" & pids+=($!); seeds+=("$s")
    else
      $fn "$s" || { echo "!! $fn（seed $s）失敗，請查看 logs/ 中對應的紀錄"; exit 1; }
    fi
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "!! $fn（seed ${seeds[$i]}）失敗，請查看 logs/ 中對應的紀錄"; fail=1; }
  done
  [ $fail -eq 0 ] || exit 1
}

# ---------------------------------------------------------------- 第 0 階段：環境檢查
stage0 () {
  echo "== 第 0 階段：單元測試與 basic strategy 基準 =="
  python -m pytest -q tests | tee logs/stage0_tests.txt
  python -m blackjack_rl.evaluate --policy basic $RULE --rounds 2000000 | tee logs/stage0_basic.txt
  echo "檢查：basic strategy 每單位下注 EV 應約在 -0.3% ~ -0.6%"
}

# ---------------------------------------------------------------- 第 1 階段：基準表
stage1 () {
  echo "== 第 1 階段：basic、DP、隨機策略在多個獨立測試 seed 上的統一評估 =="
  python -m blackjack_rl.baselines $RULE --shoe-info $SHOE --seeds ${TEST_SEEDS// /,} \
    --rounds $FINAL_ROUNDS --dp-rounds 300000 --random-rounds 100000 --paired-rounds $FINAL_ROUNDS \
    --json runs/baselines.json | tee logs/stage1_baselines.txt
  echo "檢查：DP − basic（配對）合併後應顯著為正。此步約需 1.5~2 小時"
}

# ---------------------------------------------------------------- 第 2 階段：DP 蒸餾
distill_one () {
  local s=$1
  python -m blackjack_rl.train $RULE --no-betting --shoe-info $SHOE \
    --teacher dp --pretrain-rounds $DP_ROUNDS --pretrain-updates $DISTILL_UPDATES \
    --teacher-cache runs/dp_cache_s$s.npz --steps 0 --seed $s \
    --out runs/distill_s$s > logs/stage2_distill_s$s.txt 2>&1
  echo "  distill_s$s 完成"
}
stage2 () {
  echo "== 第 2 階段：DP 蒸餾出牌頭（每個 seed 各自收集 DP 示範，約 10~15 分鐘）=="
  run_seeds distill_one
  grep -h "蒸餾" logs/stage2_distill_s*.txt
  echo "檢查：蒸餾最後 loss 應低於約 0.005"
}

# ---------------------------------------------------------------- 第 3 階段：開發 seed 快速檢查
dev_one () {
  local s=$1
  python -m blackjack_rl.paired runs/distill_s$s/last.pt --rounds 100000 --seed $DEV_SEED \
    > logs/stage3_dev_s$s.txt 2>&1
}
stage3 () {
  echo "== 第 3 階段：開發 seed 快速檢查（相對 basic，各 10 萬局）=="
  run_seeds dev_one
  cat logs/stage3_dev_s*.txt | grep "distill"
  echo "檢查：三個模型都應為正值；若有明顯為負的，先檢查該 seed 的蒸餾 loss"
}

# ---------------------------------------------------------------- 第 4 階段：出牌的最終驗證
test_play_one () {
  local s=$1
  for t in $TEST_SEEDS; do
    python -m blackjack_rl.paired runs/distill_s$s/last.pt --rounds $FINAL_ROUNDS --seed $t \
      >> logs/stage4_test_s$s.txt 2>&1
  done
}
stage4 () {
  echo "== 第 4 階段：出牌最終驗證（3 模型 × 3 測試 seed × 100 萬局配對評估）=="
  run_seeds test_play_one
  cat logs/stage4_test_s*.txt | grep "distill"
  echo "判定：9 次結果合併後信賴區間下限 > 0 即為顯著優於 basic strategy"
}

# ---------------------------------------------------------------- 第 5 階段：組合完整系統
full_one () {
  local s=$1
  python -m blackjack_rl.train $RULE --shoe-info $SHOE \
    --init-play runs/distill_s$s/last.pt --freeze-play --bet-memory 100000000 \
    --steps $BET_STEPS --eval-every 1000000 --seed $s \
    --out runs/full_s$s > logs/stage5_full_s$s.txt 2>&1
  echo "  full_s$s 完成"
}
stage5 () {
  echo "== 第 5 階段：蒸餾出牌頭（凍結）+ 訓練下注頭 =="
  run_seeds full_one
  echo "檢查：logs/stage5_full_s*.txt 末段的下注分析，平均下注應隨 true count 上升"
}

# ---------------------------------------------------------------- 第 6 階段：線性下注對照
linear_one () {
  local s=$1
  python -m blackjack_rl.bet_learn $RULE --play runs/full_s$s/last.pt \
    --rounds $BET_LINEAR_ROUNDS --out runs/full_s$s > logs/stage6_linear_s$s.txt 2>&1
  echo "  linear_s$s 完成"
}
stage6 () {
  echo "== 第 6 階段：以相同出牌策略訓練線性下注（對照組）=="
  run_seeds linear_one
}

# ---------------------------------------------------------------- 第 6b 階段：下注模型驗證
bet_eval_one () {
  local s=$1
  python -m blackjack_rl.bet_eval $RULE --play runs/full_s$s/last.pt \
    --linear runs/full_s$s/bet_linear.npz --net runs/full_s$s/bet_net.pt --head runs/full_s$s/last.pt \
    --calib-rounds 1000000 --calib-seed $((3033 + s)) --seq-rounds 1000000 --seq-seed $((4044 + s)) \
    --json runs/bet_eval_s$s.json > logs/stage6b_bet_eval_s$s.txt 2>&1
  echo "  bet_eval_s$s 完成"
}
stage6b () {
  echo "== 第 6b 階段：下注模型的校準（獨立測試牌局）與同序列下注效果 =="
  run_seeds bet_eval_one
  head -40 logs/stage6b_bet_eval_s${TRAIN_SEEDS%% *}.txt
}

# ---------------------------------------------------------------- 第 7 階段：完整系統最終評估
test_full_one () {
  local s=$1
  for t in $TEST_SEEDS; do
    python -m blackjack_rl.compare $RULE --play-model runs/full_s$s/last.pt \
      --bet-linear runs/full_s$s/bet_linear.npz --rounds $FINAL_ROUNDS --seed $t \
      >> logs/stage7_full_s$s.txt 2>&1
  done
}
stage7 () {
  echo "== 第 7 階段：完整系統比較矩陣（出牌 × 下注，100 萬局）=="
  run_seeds test_full_one
  grep -h "DQN 出牌.*DQN 下注頭" logs/stage7_full_s*.txt
  echo "判定：『DQN 出牌 + DQN 下注頭』每局報酬的信賴區間下限 > 0 即為正期望值"
}

# ---------------------------------------------------------------- 第 9 階段：資金管理
stage9 () {
  echo "== 第 9 階段：資金管理（期限內翻倍）=="
  local s1=${TRAIN_SEEDS%% *} src=runs/full_s${TRAIN_SEEDS%% *}/last.pt
  [ -f runs/bank_train.npz ] || python -m blackjack_rl.bankroll record $RULE --play $src --edge $src \
      --rounds 3000000 --seed 11 --out runs/bank_train.npz > logs/stage9_record_train.txt 2>&1 &
  [ -f runs/bank_test.npz ] || python -m blackjack_rl.bankroll record $RULE --play $src --edge $src \
      --rounds 10000000 --seed 22 --out runs/bank_test.npz > logs/stage9_record_test.txt 2>&1 &
  wait
  python -m blackjack_rl.bankroll dp --seq runs/bank_train.npz --horizon 1000 --out runs/bank_dp_h1000.npz
  python -m blackjack_rl.bankroll train --seq runs/bank_train.npz --eval-seq runs/bank_test.npz \
    --horizon 1000 --episodes 1000000 --out runs/bank_dqn_h1000 > logs/stage9_train.txt 2>&1
  python -m blackjack_rl.bankroll evaluate --seq runs/bank_test.npz --train-seq runs/bank_train.npz \
    --horizon 500 --dp runs/bank_dp_h1000.npz --dqn runs/bank_dqn_h1000/model.pt \
    --grid-starts 25,50,75 --grid-horizons 250,500,1000 --json runs/bankroll_report.json \
    | tee logs/stage9_report.txt
  python -m blackjack_rl.bankroll heatmap --horizon 500 --dqn runs/bank_dqn_h1000/model.pt \
    --dp runs/bank_dp_h1000.npz --out logs/stage9_heatmap.png
}

# ---------------------------------------------------------------- 消融實驗（選用）
ablation () {
  echo "== 消融實驗：說明各設計的必要性（各用訓練 seed 1）=="
  local common="$RULE --no-betting --shoe-info $SHOE --lr 1e-4 --lr-end 1e-5 --save-every 500000"
  # A. 從零訓練 DQN
  python -m blackjack_rl.train $common --steps 2000000 --eps-decay 1000000 \
    --seed 1 --out runs/abl_scratch > logs/abl_scratch.txt 2>&1 &
  # B. 模仿 basic strategy + RL
  python -m blackjack_rl.train $common --pretrain-rounds 200000 --pretrain-updates 10000 \
    --eps-start 0.1 --steps 2000000 --eps-decay 1000000 \
    --seed 1 --out runs/abl_imit_basic > logs/abl_imit_basic.txt 2>&1 &
  # C. DP 蒸餾 + RL 微調（沿用第 2 階段 seed 1 的快取）
  python -m blackjack_rl.train $common --teacher dp --pretrain-rounds $DP_ROUNDS \
    --pretrain-updates $DISTILL_UPDATES --teacher-cache runs/dp_cache_s1.npz \
    --anchor-weight 0.1 --eps-start 0.1 --steps 2000000 --eps-decay 1000000 \
    --seed 1 --out runs/abl_dp_rl > logs/abl_dp_rl.txt 2>&1 &
  # D. 無牌堆資訊的 DP 蒸餾（牌堆資訊本身的價值）
  python -m blackjack_rl.train $RULE --no-betting --shoe-info none \
    --teacher dp --pretrain-rounds $DP_ROUNDS --pretrain-updates $DISTILL_UPDATES \
    --teacher-cache runs/dp_cache_none_s1.npz --steps 0 \
    --seed 1 --out runs/abl_dp_noshoe > logs/abl_dp_noshoe.txt 2>&1 &
  wait
  for m in runs/abl_scratch/last.pt runs/abl_imit_basic/last.pt \
           runs/abl_dp_rl/last.pt runs/abl_dp_noshoe/last.pt runs/distill_s1/last.pt; do
    python -m blackjack_rl.paired $m --rounds 300000 --seed 2027 | tee -a logs/ablation.txt
  done
}

# ---------------------------------------------------------------- 主程式
[ $# -eq 0 ] && { sed -n '2,14p' "$0"; exit 1; }
for arg in "$@"; do
  case $arg in
    all) for k in 0 1 2 3 4 5 6 7; do stage$k; done ;;
    ablation) ablation ;;
    [0-7]|6b|9) stage$arg ;;
    *) echo "未知的階段：$arg"; exit 1 ;;
  esac
done
