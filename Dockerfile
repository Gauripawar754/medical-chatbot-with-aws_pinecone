FROM python:3.11.9-slim-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies needed for Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
 && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better cache usage
COPY requirements.txt .

# Install Python dependencies without cache
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the application
COPY . .
# Optional: create non-root user
RUN useradd -m appuser
USER appuser

# Expose the port your app runs on (change if needed)
EXPOSE 8080

# Run the application
CMD ["python", "app.py"]
