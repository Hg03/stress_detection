# .devcontainer/Dockerfile
FROM mcr.microsoft.com/devcontainers/python:1-3.10-bookworm

# Enable devcontainer features
LABEL dev.containers.featureSetVersion=1

# Set working directory
WORKDIR /workspace

# Copy your code in (this happens before postCreateCommand runs)
COPY . /workspace
