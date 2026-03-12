# SPDX-FileCopyrightText: © 2025 DSLab - Fondazione Bruno Kessler
# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- Stage: deps ----
FROM base AS deps

WORKDIR /build

# Copy only the packaging metadata first so Docker's layer cache is not
# invalidated by source-code edits.
COPY pyproject.toml README.md ./
# Create an empty src layout so the editable install resolves correctly
RUN mkdir -p src/task_inference

# Install the package in editable mode with the transformers + audio extras.
# PyTriton is installed separately because it bundles large Triton binaries.
COPY src/ src/
RUN pip3 install --no-cache-dir -e ".[transformers,audio]" \
    && pip3 install --no-cache-dir nvidia-pytriton

# ---- Stage: runtime ----
FROM deps AS runtime

WORKDIR /app

# Copy built packages forward (already installed in site-packages via editable
# install - we just need the src tree to remain resolvable)
COPY --from=deps /build/src /app/src
COPY --from=deps /build/pyproject.toml /build/README.md /app/

# Install in editable mode in the final layer so the src/ directory is found
RUN pip3 install --no-cache-dir -e ".[transformers,audio]"

# Copy the server code
COPY examples/server/server.py /app/server.py

# Triton HTTP | gRPC | Metrics
EXPOSE 8000 8001 8002

# Sensible defaults - all can be overridden via environment variables or
# docker-compose / Kubernetes env injection (see README).
ENV BACKEND=transformers \
    DEVICE=cpu \
    HTTP_PORT=8000 \
    GRPC_PORT=8001 \
    METRICS_PORT=8002

ENTRYPOINT ["python3", "/app/server.py"]
