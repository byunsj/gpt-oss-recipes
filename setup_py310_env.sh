#!/bin/bash
# py310 환경에 추가 패키지 설치 스크립트

echo "🔧 py310 환경에 필요한 패키지 추가 설치 중..."

# 기존 py310 환경 활성화
source /local/scratch/gbyun/python_envs/py310/bin/activate

echo "📦 현재 환경: $(which python)"
echo "📦 Python 버전: $(python --version)"

# 필요한 패키지들 확인 및 설치
echo "🔍 필요한 패키지들 확인 중..."

# jinja2 (GPT-OSS chat template용)
python -c "import jinja2; print('✅ jinja2 already installed')" 2>/dev/null || pip install jinja2

# accelerate (분산 훈련용)
python -c "import accelerate; print('✅ accelerate already installed')" 2>/dev/null || pip install accelerate

# 최신 transformers, trl, peft 확인
echo "🔄 패키지 버전 확인 중..."
python -c "
import transformers, trl, peft
print(f'transformers: {transformers.__version__}')
print(f'trl: {trl.__version__}') 
print(f'peft: {peft.__version__}')
"

echo "✅ py310 환경 설정 완료!"
echo "💡 이제 다음 명령으로 훈련 시작:"
echo "   sbatch run_user_crisis_training.sh"