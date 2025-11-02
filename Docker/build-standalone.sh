#!/bin/bash
set -e

echo " Building IVIM Fitting for STANDALONE..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

docker build \
  --build-arg ENV_TYPE=standalone \
  -t ivim-fitting:standalone \
  -f Docker/Dockerfile.unified \
  .

echo ""
echo " Build successful!"
echo " Image: ivim-fitting:standalone"
docker images | grep "ivim-fitting.*standalone"
