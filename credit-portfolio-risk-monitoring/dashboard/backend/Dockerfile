FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=65536
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
EXPOSE 8000
# One worker: each extra worker holds its own copy of pandas, SciPy and the cached data.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
