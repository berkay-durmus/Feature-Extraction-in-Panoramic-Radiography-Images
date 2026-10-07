# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .


FROM python:3.12-slim AS runtime

ARG VERSION=dev
LABEL org.opencontainers.image.title="panoramic-features" \
      org.opencontainers.image.description="Third-molar and mandibular canal feature extraction from panoramic radiographs" \
      org.opencontainers.image.source="https://github.com/berkay-durmus/Feature-Extraction-in-Panoramic-Radiography-Images" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN useradd --system --uid 10001 --no-create-home app \
    && mkdir /data /output \
    && chown app /output

COPY --from=builder /opt/venv /opt/venv

USER app
WORKDIR /output
VOLUME ["/data", "/output"]

ENTRYPOINT ["panoramic-features"]
CMD ["/data", "-o", "/output/FeatureList.xlsx"]
