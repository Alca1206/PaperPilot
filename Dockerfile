# PaperPilot app image.
# Ollama is NOT inside this image: the app talks to Ollama over HTTP
# (on your computer, or in a second container; see docker-compose.yml).

FROM python:3.12-slim

# No .pyc files, unbuffered logs, no pip cache inside the image
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Streamlit: listen on all interfaces inside the container, no usage statistics
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    # Where PaperPilot keeps uploads, the vector store and exports (mounted as a volume)
    PAPERPILOT_DATA_DIR=/app/data

WORKDIR /app

# 1) Dependencies first: this layer is cached until requirements.txt changes
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2) Then the code (changes often, rebuilds in seconds)
COPY app.py pytest.ini ./
COPY paperpilot ./paperpilot
COPY tests ./tests

# Run as a normal user, not root
RUN useradd --create-home --uid 1000 paperpilot \
    && mkdir -p /app/data \
    && chown -R paperpilot:paperpilot /app
USER paperpilot

EXPOSE 8501

# Docker marks the container "healthy" once Streamlit answers
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py"]