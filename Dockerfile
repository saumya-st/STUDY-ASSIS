# NOTE: this Dockerfile has not been built or run yet (Docker was unavailable when it
# was written). Treat it as a starting point and verify locally before relying on it.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STUDY_DB_PATH=/data/study_coach.db

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY smart_study.py .
COPY study_coach ./study_coach

RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8501

CMD ["streamlit", "run", "smart_study.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"]
