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

# PyTorch 2.4.0 + CUDA 12.1.
# --extra-index-url so PyPI stays in the search path for transitive deps
# (fsspec, sympy, etc.) - --index-url alone replaces PyPI and the build fails.
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.0

# SPURS inference-only deps (mirrors requirements.inference.txt on the beta branch).
RUN pip install --no-cache-dir \
    "omegaconf>=2.3" \
    "huggingface_hub>=0.20" \
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
RUN python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='cyclization9/SPURS', filename='spurs/.hydra/config.yaml'); \
    hf_hub_download(repo_id='cyclization9/SPURS', filename='spurs/checkpoints/best.ckpt'); \
    hf_hub_download(repo_id='cyclization9/SPURS', filename='spurs_multi/.hydra/config.yaml'); \
    hf_hub_download(repo_id='cyclization9/SPURS', filename='spurs_multi/checkpoints/best.ckpt'); \
    print('SPURS checkpoints cached.')"

RUN mkdir -p inputs out && chmod -R 777 /app
