FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Runtime shared libs required by python wheels (notably osmium).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Build-only dependency set for deterministic dataset reproduction.
RUN pip install \
    "numpy==2.2.6" \
    "pandas==2.3.2" \
    "shapely==2.1.1" \
    "pyproj==3.7.2" \
    "pyogrio==0.11.1" \
    "geopandas==1.1.1" \
    "osmium==4.1.1"

# Default entrypoint is generic build dispatcher.
ENTRYPOINT ["python", "/app/build.py"]
