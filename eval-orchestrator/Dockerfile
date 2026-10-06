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

# Clone evaluerBench into /opt so it doesn't get overwritten by the /app volume mount.
# EVALUERBENCH_REF is a branch, tag, or commit SHA; pin a SHA for reproducible runs:
#   EVALUERBENCH_REF=<sha> docker compose build eval-worker
ARG EVALUERBENCH_REF=main
RUN git clone https://github.com/joshoney/evaluerBench.git /opt/evaluerBench     && git -C /opt/evaluerBench checkout --detach "$EVALUERBENCH_REF"     && git -C /opt/evaluerBench log -1 --format="evaluerBench at %H"
# Install Node dependencies at build time, not inside the eval activity's timeout
RUN cd /opt/evaluerBench && npm ci --no-audit --no-fund
ENV PYTHONPATH="/opt"

# Application code is bind-mounted at /app by docker-compose (not copied into the image).
# A branch ref like main is resolved at build time; pick up new commits with:
#   docker compose build --no-cache eval-worker
# Run the worker script as the main process
CMD ["python", "worker.py"]