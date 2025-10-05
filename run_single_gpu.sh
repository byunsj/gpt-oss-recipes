#!/bin/bash
#SBATCH --job-name=user-crisis-120b-qlora-single
#SBATCH --output=user_crisis_single_gpu.out
#SBATCH --gres=gpu:1 
#SBATCH --account=csaiadv
#SBATCH --mem=200G
#SBATCH --cpus-per-task=32

echo "=========================================="
echo "User Crisis Classification Training (Single GPU)"
echo "Model: lmsys/gpt-oss-120b-bf16"
echo "Method: QLoRA Single GPU"
echo "=========================================="

# 환경 활성화
source /local/scratch/gbyun/python_envs/oss120/bin/activate

# 환경 변수
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="/local/scratch/gbyun/.triton_cache"

# HF 토큰 확인
if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "⚠️  Please set HUGGING_FACE_HUB_TOKEN"
    exit 1
fi

echo "🚀 Starting single GPU training..."

# 단일 GPU로 실행 - YAML 설정 파일 사용
CUDA_VISIBLE_DEVICES=0 python crisis_sft.py configs/user_crisis_qlora_120b.yaml

echo "✅ Training Complete!"