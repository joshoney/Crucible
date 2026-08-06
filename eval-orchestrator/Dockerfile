FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Node.js and Git for evaluerBench
RUN apt-get update && apt-get install -y curl git \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone evaluerBench into /opt so it doesn't get overwritten by the /app volume mount
RUN git clone https://github.com/joshoney/evaluerBench.git /opt/evaluerBench
ENV PYTHONPATH="/opt"

# The application code (and evaluerBench) will be mounted via docker-compose volumes
# Run the worker script as the main process
CMD ["python", "worker.py"]