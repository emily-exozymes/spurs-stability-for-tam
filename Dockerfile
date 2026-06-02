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

# Pre-fetch both SPURS checkpoints from HuggingFace at build time. Download
# each specific file (snapshot_download has been observed to skip hidden
# .hydra/ directories in some versions). Copy them to fixed local paths so
# runtime never touches the HF cache or network.
RUN python -c "import shutil, os; \
    from huggingface_hub import hf_hub_download; \
    files = [ \
        ('spurs',       '.hydra/config.yaml'), \
        ('spurs',       'checkpoints/best.ckpt'), \
        ('spurs_multi', '.hydra/config.yaml'), \
        ('spurs_multi', 'checkpoints/best.ckpt'), \
    ]; \
    [os.makedirs(f'/opt/spurs_checkpoints/{s}/{os.path.dirname(f)}', exist_ok=True) for s, f in files]; \
    [shutil.copy(hf_hub_download(repo_id='cyclization9/SPURS', filename=f'{s}/{f}'), f'/opt/spurs_checkpoints/{s}/{f}') for s, f in files]; \
    print('SPURS checkpoints copied to /opt/spurs_checkpoints/'); \
    import subprocess; subprocess.run(['find', '/opt/spurs_checkpoints', '-type', 'f'])"

# Make checkpoints readable by any UID Tamarind happens to use at runtime.
RUN chmod -R a+rX /opt/spurs_checkpoints

# SPURS calls fair-esm's load_model_and_alphabet_hub() internally, which
# downloads the ESM2-650M backbone from dl.fbaipublicfiles.com to
# $TORCH_HOME/hub/checkpoints/. Pre-fetch both the model and its
# contact-regression weights so runtime never needs network. Set
# TORCH_HOME at runtime via envVars in config.json to point here.
RUN mkdir -p /opt/torch_hub/hub/checkpoints && \
    cd /opt/torch_hub/hub/checkpoints && \
    python -c "import urllib.request, os; \
    urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt', 'esm2_t33_650M_UR50D.pt'); \
    urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt', 'esm2_t33_650M_UR50D-contact-regression.pt'); \
    print('ESM2-650M cached at /opt/torch_hub/hub/checkpoints/'); \
    import subprocess; subprocess.run(['ls', '-la', '/opt/torch_hub/hub/checkpoints/'])"

RUN chmod -R a+rX /opt/torch_hub

# Force offline mode at runtime so HF never attempts a HEAD/network call.
# This is the critical fix - without it, cached files still trigger HEAD
# requests to check for updates, which fail when there's no DNS.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

RUN mkdir -p inputs out && chmod -R 777 /app
