#!/bin/bash
set -e

echo " Building IVIM Fitting for KAAPANA..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

docker build \
  --build-arg BASE_IMAGE=local-only/base-python-cpu:latest \
  --build-arg ENV_TYPE=kaapana \
  -t ivim-fitting:kaapana \
  -f Docker/Dockerfile.unified \
  .

echo ""
echo " Build successful!"
echo " Image: ivim-fitting:kaapana"
docker images | grep "ivim-fitting.*kaapana"
