FROM python:3.10-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    zsh \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Create non-root user with zsh
RUN useradd -m -s /bin/zsh devuser
USER devuser

# Install Oh My Zsh
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

WORKDIR /workspace

# Copy project files
COPY --chown=devuser:devuser . .

# Set git safe directory
RUN git config --global --add safe.directory /workspace

# Create and activate virtual environment
RUN uv venv .venv && \
    uv pip install -e ".[dev]" --python .venv/bin/python

# Configure zsh to auto-activate venv
RUN echo 'source /workspace/.venv/bin/activate' >> ~/.zshrc && \
    echo 'echo "Python virtual environment activated"' >> ~/.zshrc

# Set environment variables for activated venv
ENV VIRTUAL_ENV="/workspace/.venv"
ENV PATH="/workspace/.venv/bin:$PATH"

CMD ["fastapi", "dev", "src/stress_detection/scripts/infer.py", "--port", "8080"]