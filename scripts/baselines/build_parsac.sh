#!/usr/bin/env bash
# Fetch PARSAC (Intel Labs, Apache-2.0), apply parsac.patch and build its C++ annealer at -O3.
#
# Usage:   bash scripts/baselines/build_parsac.sh
# Result:  third_party/parsac/ (source at the pinned commit + patch) and
#          third_party/parsac/build/ca_sa_o3.so, the module trinity_baselines.classical.parsac loads.
# Re-running is safe: an existing checkout is reused and the patch is applied once.
# TRINITY_PARSAC / TRINITY_PARSAC_BUILD override the source and build directories.
set -euo pipefail

PARSAC_URL="https://github.com/IntelLabs/parsac.git"
PARSAC_COMMIT="bff459ba7cc4b71d3d78f016e146f9a71f5b62ab"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PATCH="${SCRIPT_DIR}/parsac.patch"
PARSAC_DIR="${TRINITY_PARSAC:-${REPO_ROOT}/third_party/parsac}"
export TRINITY_PARSAC="${PARSAC_DIR}"

if [ ! -d "${PARSAC_DIR}/.git" ]; then
    echo "cloning ${PARSAC_URL} into ${PARSAC_DIR}"
    mkdir -p "$(dirname "${PARSAC_DIR}")"
    git clone "${PARSAC_URL}" "${PARSAC_DIR}"
fi

if git -C "${PARSAC_DIR}" apply --reverse --check "${PATCH}" 2>/dev/null; then
    echo "parsac.patch already applied"
else
    git -C "${PARSAC_DIR}" fetch --quiet origin
    git -C "${PARSAC_DIR}" checkout --quiet "${PARSAC_COMMIT}"
    git -C "${PARSAC_DIR}" apply "${PATCH}"
    echo "checked out ${PARSAC_COMMIT} and applied parsac.patch"
fi

echo "building the -O3 extension (torch.utils.cpp_extension)"
python -c "from trinity_baselines.classical.parsac import engine; engine(); print('PARSAC engine ready')"
