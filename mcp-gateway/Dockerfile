FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000

# Run the ASGI app for HTTP/SSE transport
CMD ["uvicorn", "main:mcp.app", "--host", "0.0.0.0", "--port", "8000"]