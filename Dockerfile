# https://huggingface.co/docs/hub/spaces-sdks-docker
FROM python:3.11-slim

# buat user non-root (direkomendasikan oleh Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# install deps
COPY --chown=user ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# copy kode & artefak model
COPY --chown=user . /app

# Hugging Face Spaces memakai port 7860 secara default
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
