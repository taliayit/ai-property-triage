# ---- builder: compiles llama-cpp-python (needs a C/C++ toolchain + cmake,
# which the slim runtime image below deliberately doesn't carry) ----
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# ---- runtime: slim image, only the prebuilt wheels + app code ----
FROM python:3.11-slim

# libgomp1: OpenMP runtime llama.cpp needs for multi-threaded CPU inference.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY config.py main.py seed_data.py ./

# Bundled synthetic listings are seeded at build time so the container
# serves /query immediately. The GGUF model is NOT baked in here - it's
# hundreds of MB to several GB, so it's mounted at runtime instead (see
# README for the `docker run -v` command).
RUN python seed_data.py
RUN mkdir -p /app/models

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
