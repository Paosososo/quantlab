# ---------------------------------------------------------------------------
# quantlab application image: API, dashboard and CLI.
#
# Multi-stage.  The builder compiles wheels; the runtime copies them in and
# carries no compiler.  That drops roughly 400MB and removes a toolchain from
# the attack surface of a running service.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

# Install into a virtualenv so the runtime stage can copy one self-contained
# directory instead of picking site-packages apart.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel \
    && pip install ".[api,dashboard]"


FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    QUANTLAB_LOG_JSON=true

# libpq5 only: the client library, not the build headers.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 quantlab

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=quantlab:quantlab alembic.ini ./
COPY --chown=quantlab:quantlab migrations ./migrations
COPY --chown=quantlab:quantlab scripts ./scripts
RUN mkdir -p /app/data && chown -R quantlab:quantlab /app/data

USER quantlab

# Deliberately no ``COPY src`` and no ``PYTHONPATH=/app/src``.  An earlier
# version had both: the package was pip-installed into /opt/venv *and* the
# source tree was copied to /app/src with PYTHONPATH pointing at it.  The image
# then held two copies of the code and the copy that actually executed was the
# unpackaged one, which shadowed the installed distribution.  That breaks the
# thing an image is supposed to guarantee -- that what runs is what was built --
# because /app/src carries no version metadata, no entry points and no record of
# how it was installed.  ``pip install .`` in the builder stage is now the only
# path into the image, so ``importlib.metadata.version("quantlab")`` and the
# running code cannot disagree.  Alembic and Streamlit both resolve the package
# from the venv (see quantlab/dashboard/__main__.py for how Streamlit does it).

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "quantlab.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
