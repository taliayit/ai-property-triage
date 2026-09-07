FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py model.py dataset.py main.py ./
COPY models/room_condition_model.pt models/room_condition_model.pt

# Pre-download the ImageNet-pretrained EfficientNet-B0 backbone weights at
# build time (torchvision fetches them from download.pytorch.org on first
# use otherwise), so the container works without internet access at
# runtime - same reasoning as rag-service's build-time seed step.
RUN python -c "from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights; efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8003

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
