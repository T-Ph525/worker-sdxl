# Base image
FROM runpod/base:0.4.2-cuda11.8.0

# Environment variables
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies
COPY builder/requirements.txt /requirements.txt
RUN apt update && \
    apt install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/* && \
    python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --prefer-binary --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Cache models
COPY builder/cache_models.py /app/cache_models.py
RUN python3.11 /app/cache_models.py

# Add source files
ADD src /app/

# Set execution command
CMD ["python3.11", "-u", "/app/rp_handler.py"]

