FROM python:3.10-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Install development tools
RUN pip install ruff black mypy

# Create a non-root user
RUN useradd -m -s /bin/bash devuser
USER devuser
WORKDIR /home/devuser/app

# Copy project files
COPY --chown=devuser:devuser . .

# Create virtual environment and install dependencies
RUN uv venv .venv && \
    uv pip install -e ".[dev]" --python .venv/bin/python

# Add .venv to PATH and create activation alias
RUN echo 'export PATH="/home/devuser/app/.venv/bin:$PATH"' >> ~/.bashrc && \
    echo 'alias activate="source /home/devuser/app/.venv/bin/activate"' >> ~/.bashrc

# Set environment variables
ENV PATH="/home/devuser/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/home/devuser/app/.venv"

# Default command
CMD ["/bin/bash"]