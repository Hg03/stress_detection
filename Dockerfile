FROM python:3.10-slim-bookworm

# Install system dependencies including zsh
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    zsh \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Install development tools
RUN pip install ruff black mypy

# Create a non-root user with zsh as default shell
RUN useradd -m -s /bin/zsh devuser
USER devuser

# Install Oh My Zsh
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

WORKDIR /home/devuser/code

# Copy project files
COPY --chown=devuser:devuser . .

# Create virtual environment and install dependencies
RUN uv venv .venv && \
    uv pip install -e ".[dev]" --python .venv/bin/python

# Add .venv to PATH and create activation alias in zsh config
RUN echo 'export PATH="/home/devuser/code/.venv/bin:$PATH"' >> ~/.zshrc && \
    echo 'alias activate="source /home/devuser/code/.venv/bin/activate"' >> ~/.zshrc

# Set environment variables
ENV PATH="/home/devuser/code/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/home/devuser/code/.venv"

# Default command
CMD ["/bin/zsh"]
