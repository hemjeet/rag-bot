FROM python:3.11-slim

# Prevent Python from writing .pyc files to save space
ENV PYTHONDONTWRITEBYTECODE=1
# Ensure logs are flushed immediately
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy and install requirements first (leverages Docker cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user for security
RUN useradd -m appuser
USER appuser

# Copy application code with non-root ownership
COPY --chown=appuser:appuser . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
