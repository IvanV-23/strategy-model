FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    cmake \
    ninja-build \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir numpy gymnasium

# Create working directory
WORKDIR /workspace

# Copy project files
COPY . .

# Install opencode using official installer
RUN curl -fsSL https://opencode.ai/install | bash
ENV PATH="/root/.opencode/bin:$PATH"

# Set default command
CMD ["/bin/bash"]