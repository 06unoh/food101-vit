# PyTorch 2.6.0 + CUDA 12.4 기반
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# 작업 디렉토리 설정
WORKDIR /app

# pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip

# 정확한 버전으로 패키지 설치
RUN pip install --no-cache-dir torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install --no-cache-dir matplotlib==3.10.0 numpy==2.0.2

# 현재 프로젝트 전체 복사
COPY . .

# 메인 실행 명령 → python main.py 실행
CMD ["python", "main.py"]
