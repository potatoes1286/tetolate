# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.11
ARG UV_VERSION=0.11.24
ARG IMAGEMAGICK_VERSION=7.1.2-13
ARG IMAGEMAGICK_SHA256=3617bffe497690ffe5b731227d026db1150e138ddb129481a1e202442e558512

FROM python:${PYTHON_VERSION}-slim-bookworm AS imagemagick-builder
ARG IMAGEMAGICK_VERSION
ARG IMAGEMAGICK_SHA256

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        libfontconfig1-dev \
        libfreetype6-dev \
        libheif-dev \
        libjpeg62-turbo-dev \
        libjxl-dev \
        liblcms2-dev \
        libpng-dev \
        libtiff-dev \
        libwebp-dev \
        libxml2-dev \
        libzip-dev \
        libzstd-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/imagemagick
RUN curl -fsSL \
        "https://github.com/ImageMagick/ImageMagick/archive/refs/tags/${IMAGEMAGICK_VERSION}.tar.gz" \
        -o source.tar.gz \
    && echo "${IMAGEMAGICK_SHA256}  source.tar.gz" | sha256sum -c - \
    && tar -xzf source.tar.gz --strip-components=1 \
    && ./configure \
        --prefix=/opt/imagemagick \
        --disable-static \
        --with-modules=no \
        --without-perl \
        --without-x \
        --with-fontconfig=yes \
        --with-freetype=yes \
        --with-heic=yes \
        --with-jpeg=yes \
        --with-jxl=yes \
        --with-lcms=yes \
        --with-png=yes \
        --with-tiff=yes \
        --with-webp=yes \
    && make -j"$(nproc)" \
    && make install \
    && rm -rf \
        /opt/imagemagick/include \
        /opt/imagemagick/lib/pkgconfig \
        /opt/imagemagick/share/doc \
        /opt/imagemagick/share/man \
    && /opt/imagemagick/bin/magick -version

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:${PYTHON_VERSION}-slim-bookworm AS python-builder-base
COPY --from=uv /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=0 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./

FROM python-builder-base AS app-dependency-builder
# PaddleX declares the GUI-enabled wheel, but tetolate never opens OpenCV windows.
# Replace it before copying the environment into the runtime image. PaddleX
# checks the original distribution name at runtime, so retain only its small
# dist-info metadata after removing the GUI wheel's code and shared libraries.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project --extra inpaint --extra ocr \
    && cp -a \
        /app/.venv/lib/python3.11/site-packages/opencv_contrib_python-4.10.0.84.dist-info \
        /tmp/opencv-contrib-dist-info \
    && uv pip uninstall --python /app/.venv/bin/python opencv-contrib-python \
    && uv pip install \
        --python /app/.venv/bin/python \
        opencv-contrib-python-headless==4.10.0.84 \
    && cp -a \
        /tmp/opencv-contrib-dist-info \
        /app/.venv/lib/python3.11/site-packages/opencv_contrib_python-4.10.0.84.dist-info

FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime-base

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/imagemagick/bin:/app/.venv/bin:$PATH \
    LD_LIBRARY_PATH=/opt/imagemagick/lib \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PADDLE_PDX_CACHE_HOME=/data/cache/paddlex \
    PADDLEX_HOME=/data/cache/paddlex \
    PADDLEX_TEMP_DIR=/data/cache/paddlex/temp \
    HF_HOME=/data/cache/huggingface \
    TORCH_HOME=/data/cache/torch \
    XDG_CACHE_HOME=/data/cache \
    TETOLATE_DATA_DIR=/data

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        fontconfig \
        fonts-dejavu-core \
        gosu \
        libfontconfig1 \
        libfreetype6 \
        libglib2.0-0 \
        libgomp1 \
        libheif1 \
        libjpeg62-turbo \
        libjxl0.7 \
        liblcms2-2 \
        libpng16-16 \
        libtiff6 \
        libwebp7 \
        libwebpdemux2 \
        libwebpmux3 \
        libxml2 \
        libzip4 \
        libzstd1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 tetolate \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash tetolate

COPY --from=imagemagick-builder /opt/imagemagick /opt/imagemagick
COPY --from=app-dependency-builder /app/.venv /app/.venv

WORKDIR /app
COPY *.py ./
COPY web_editor/ ./web_editor/
COPY data/config /app/data/config
COPY data/prompts /app/data/prompts
COPY docker ./docker

RUN magick -version \
    && magick identify -list format | grep -Eq '^ *JXL.*rw' \
    && magick identify -list format | grep -Eq '^ *WEBP.*rw' \
    && /app/.venv/bin/python -c "import translate_cbz, web_app" \
    && /app/.venv/bin/python -c "import cv2, paddle, paddleocr" \
    && /app/.venv/bin/python -c "import lama_inpaint, torch; print(torch.__version__)"

FROM runtime-base AS tests
COPY tests ./tests
RUN python -m unittest tests.test_regressions \
    && touch /tmp/tests-passed

FROM runtime-base AS runtime
COPY --from=tests /tmp/tests-passed /opt/tetolate-tests-passed

VOLUME ["/data"]
EXPOSE 8088

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["/app/.venv/bin/python", "/app/web_app.py", "--config", "/data/config/web_config.json"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8088/admin', timeout=3).read(1)"]
