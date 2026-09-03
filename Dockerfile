# SOC Next-Best-Action Triage Agent -- application image.
#
# This image contains ONLY the agent (Python + uv-managed deps). It never bundles
# the GLM-5.2 model. It runs in two modes:
#   * demo/mock  -- fully offline, deterministic (default CMD, no network/GPU).
#   * real       -- talks to an OpenAI-compatible vLLM endpoint via LLM_BASE_URL.
#
# See docker-compose.yml for both wirings.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# uv: copy (not hardlink) into the image layer, compile bytecode, never try to
# download a managed Python (the base image already ships 3.12).
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Dependency layer -- cached until pyproject.toml / uv.lock change.
#    This is a "virtual" project (no build backend), so --no-install-project
#    just resolves + installs the locked deps into /app/.venv.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 2) Application source. The package is imported from the working directory
#    (pythonpath), so `uv run python -m soc_triage.*` finds it without a build.
COPY soc_triage ./soc_triage

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

# ENTRYPOINT fixes the module; CMD supplies the args. Override CMD to change the
# run (e.g. drop --mock to hit a real endpoint, or run the eval module).
ENTRYPOINT ["uv", "run", "--no-sync", "python", "-m", "soc_triage.demo"]
CMD ["--mock"]
