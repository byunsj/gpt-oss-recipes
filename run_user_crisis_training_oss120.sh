#!/bin/bash
#SBATCH --job-name=user-crisis-120b-qlora
#SBATCH --output=user_crisis_gpt_oss_120b_qlora.out
#SBATCH --gres=gpu:4 
#SBATCH --account=csaiadv
#SBATCH --mem=562G
#SBATCH --cpus-per-task=128

echo "=========================================="
echo "User Crisis Classification Training"
echo "Model: lmsys/gpt-oss-120b-bf16"
echo "Method: QLoRA with OSS120 Environment"
echo "=========================================="
echo ""

# 새로운 oss120 환경 활성화
source /local/scratch/gbyun/python_envs/oss120/bin/activate

# 환경 변수 설정
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="/local/scratch/gbyun/.triton_cache"

# Hugging Face 토큰 확인
if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "⚠️  Please set HUGGING_FACE_HUB_TOKEN environment variable"
    echo "   export HUGGING_FACE_HUB_TOKEN=\"your_token_here\""
    exit 1
fi
echo "🔑 HF Token configured"
echo ""

# 패키지 버전 확인
echo "📦 Checking package versions in oss120 environment..."
python -c "
import transformers, trl, peft
print(f'transformers: {transformers.__version__}')
print(f'trl: {trl.__version__}') 
print(f'peft: {peft.__version__}')
print(f'Python path: {__import__('sys').executable}')
"
echo ""

# 시작 시간 기록
START_TIME=$(date +%s)
echo "🕐 Start time: $(date)"
echo ""

# 데이터 준비
echo "📂 Preparing user data..."
python prepare_user_data.py
echo ""

# 학습 실행
echo "🚀 Starting training with updated packages..."
echo ""

# Accelerate를 사용한 분산 학습 (4 GPU) - YAML 설정 파일 사용
accelerate launch \
    --config_file configs/zero3.yaml \
    crisis_sft.py \
    configs/user_crisis_qlora_120b.yaml

echo ""
echo "✅ Training Finished!"

# 종료 시간 계산
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "🕐 End time: $(date)"
echo "⏱️  Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""

# 결과 확인
if [ -d "./user_crisis_gpt_oss_120b_qlora" ]; then
    echo "📁 Output directory created successfully"
    ls -la ./user_crisis_gpt_oss_120b_qlora/
else
    echo "❌ Output directory not found - training may have failed"
fi

echo "=========================================="
echo "Training Complete with OSS120 Environment!"
echo "=========================================="