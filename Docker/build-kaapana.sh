#!/bin/bash
set -e

echo "📦 Building IVIM Fitting for KAAPANA (multi-stage)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

docker build \
  --target kaapana \
  --build-arg BASE_IMAGE=local-only/base-python-cpu:latest \
  -t ivim-fitting:kaapana \
  -f Docker/Dockerfile.unified \
  .

echo ""
echo "Build successful!"
echo "Image: ivim-fitting:kaapana"
docker images | grep "ivim-fitting.*kaapana"
