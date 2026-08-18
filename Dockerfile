# Base Python Image
FROM python:3.11-slim

# Working directory
WORKDIR /app

# Prevent bytecode & buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader vader_lexicon

# Copy source code
COPY . .

# Default command: run full pipeline CLI
ENTRYPOINT ["python", "main.py"]
CMD ["--samples", "600", "--clusters", "3", "--output_dir", "./reports"]
