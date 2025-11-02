# Dockerfile Consolidation - Issue #115 Task #4

## Overview
This document describes the consolidation of two separate Dockerfiles into a single unified Dockerfile that supports both standalone and Kaapana deployment.

## Problem
Previously maintained two separate Dockerfiles:
1. `Docker/Dockerfile` - Standalone IVIM fitting
2. `kaapana_ivim_osipi/.../Dockerfile` - Kaapana deployment

This created:
- Code duplication
- Maintenance overhead
- Inconsistent updates

## Solution
Created `Docker/Dockerfile.unified` that uses Docker build arguments to support both scenarios.

## Build Arguments

| Argument | Default | Options | Purpose |
|----------|---------|---------|---------|
| `BASE_IMAGE` | `python:3.11-slim` | Any valid Docker image | Choose base image |
| `ENV_TYPE` | `standalone` | `standalone` or `kaapana` | Select environment |

## Building

### Standalone Build

Method 1: Using build script
./Docker/build-standalone.sh

Method 2: Direct docker command
docker build
--build-arg ENV_TYPE=standalone
-t ivim-fitting:standalone
-f Docker/Dockerfile.unified .
### Kaapana Build
Method 1: Using build script
./Docker/build-kaapana.sh

Method 2: Direct docker command
docker build
--build-arg BASE_IMAGE=local-only/base-python-cpu:latest
--build-arg ENV_TYPE=kaapana
-t ivim-fitting:kaapana
-f Docker/Dockerfile.unified .

## Testing Completed

###  Local Testing (Docker Desktop - Windows)
- [x] Standalone build succeeds (13.2GB, ID: 75a626d42f0b)
- [x] Kaapana build succeeds (14GB, ID: f662fb55bfee)
- [x] Both conditional logic statements work correctly
- [x] File copying and dependencies install without errors
- [x] Build scripts execute correctly

###  Integration Testing (Requires Kaapana Platform)
- [ ] Runtime with actual `local-only/base-python-cpu:latest` base image
- [ ] Volume mounts work (`OPERATOR_IN_DIR`, `OPERATOR_OUT_DIR`)
- [ ] MinIO data integration
- [ ] Airflow DAG execution in Kaapana UI
- [ ] Workflow submission through web interface

## Files Modified/Added

| File | Change | Type |
|------|--------|------|
| `Docker/Dockerfile.unified` | New unified Dockerfile |  New |
| `Docker/build-standalone.sh` | Build script for standalone |  New |
| `Docker/build-kaapana.sh` | Build script for Kaapana |  New |
| `Docker/DOCKERFILE_MERGE.md` | This documentation |  New |

## Notes for Reviewers

### What I Could Test
 Docker build process for both environments
 Image creation and basic structure
 Build argument passing
 Entry point configuration

### What Needs Your Testing
 Runtime behavior in actual Kaapana deployment
 Compatibility with Kaapana base image
 Workflow execution through Kaapana UI
 Data pipeline with MinIO

## Related Issues
- Addresses #115 (Task #4)
- Related to PR #112
