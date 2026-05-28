FROM mambaorg/micromamba:1.5.8

USER root
WORKDIR /app

RUN micromamba install -y -n base -c conda-forge \
    python=3.11 \
    "setuptools<81" \
    pip \
    git \
    && micromamba clean -a -y

ENV PATH=/opt/conda/bin:$PATH

# Pin the HF cache to a predictable, world-readable location so runtime
# (which may execute as a non-root user on Tamarind) can still find weights.
ENV HF_HOME=/opt/hf_cache
ENV HUGGINGFACE_HUB_CACHE=/opt/hf_cache
ENV TRANSFORMERS_CACHE=/opt/hf_cache

# PyTorch 2.4.0 + CUDA 12.1.
# --extra-index-url so PyPI stays in the search path for transitive deps
# (fsspec, sympy, etc.) - --index-url alone replaces PyPI and the build fails.
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.0

# SPURS inference-only deps (mirrors requirements.inference.txt on the beta branch).
# huggingface_hub pinned <0.26 to avoid the closed-httpx-client retry bug
# (newer versions reuse a closed httpx client on retry and crash).
RUN pip install --no-cache-dir \
    "omegaconf>=2.3" \
    "huggingface_hub>=0.20,<0.26" \
    "fair-esm>=2.0.0" \
    "biopython>=1.79" \
    "einops>=0.6" \
    "e3nn>=0.5.6" \
    "fairscale>=0.4.13" \
    "numpy>=1.24,<2" \
    pandas \
    tqdm \
    pyyaml

# Pre-fetch both SPURS checkpoints from HuggingFace at build time.
# Tamarind runtime containers have no internet, so models MUST be baked in.
# Using snapshot_download grabs the whole repo in one shot (config + ckpts +
# any tokenizer/aux files) and is more robust than per-file hf_hub_download.
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='cyclization9/SPURS', cache_dir='/opt/hf_cache'); \
    print('SPURS snapshot cached to /opt/hf_cache')"

# Make the cache readable by any UID Tamarind happens to use at runtime.
RUN chmod -R a+rX /opt/hf_cache

# Force offline mode at runtime so HF never attempts a HEAD/network call.
# This is the critical fix — without it, cached files still trigger HEAD
# requests to check for updates, which fail when there's no DNS.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

RUN mkdir -p inputs out && chmod -R 777 /app
