FROM image.ppinfra.com/prod-gpucloudpublic/pytorch:2.7.1-cuda12.8

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    PYTHONPATH=/app/src

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-cloud.txt pyproject.toml ./
COPY src ./src
RUN python -m pip install --no-cache-dir -r requirements-cloud.txt \
    && python -m pip install --no-cache-dir --no-deps .

COPY . .
COPY --from=runtime-assets /models ./models
RUN python tools/verify_install.py

CMD ["python", "-m", "netcast_tennisvision.cloud.worker"]
