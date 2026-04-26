FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    SUSPECT_X_DESCRIPTIONS=/app/descriptions

WORKDIR /app

# Server-only deps (the trainer is NOT installed inside the env container).
RUN pip install --no-cache-dir \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.27" \
        "pydantic>=2.6"

# Copy the server package and the crime descriptions into the image.
COPY server /app/server
COPY descriptions /app/descriptions

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
