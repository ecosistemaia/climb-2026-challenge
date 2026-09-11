#!/usr/bin/env bash
#
# Populate weights/ from its source path before `docker build`.
#
# Only the ColonStreamSfSNet checkpoint needs this: it isn't downloadable (a
# private trained checkpoint, not a public hub weight), so it has to be copied
# into the build context ahead of time. RAFT's weights ARE public and are
# fetched during `docker build` itself instead (network is available at build
# time per the contract; --network=none only applies at run time) -- see the
# Dockerfile's warm-up RUN step.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

# v10: exp_9/seed42_20ep -- exp_7/l1's exact config + seed, trained 8->20 epochs.
# Lowest OPW / least temporal flicker of 19 ColonStreamSfSNet variants (exp_17);
# probe submission for whether that consistency axis transfers to Synapse
# (exp_52 local: 4-clip mean ATE -6.2%, MIXED). See docs/submission-log.md.
CKPT_SRC="${COLONSTREAM_CKPT_SRC:-/data/jpolo/depthPoseEstimation/ColonStreamSfSNet/checkpoints/experiments_640x480/exp_9/results/seed42_20ep/best_model.pth}"
CKPT_DST="${REPO_ROOT}/weights/colonstreamsfsnet/best_model.pth"
CKPT_SHA256="754c3b986bb2ca2499bde9c127a3ce30c5aa363b5e2fce3683c5f833fcd76b02"

mkdir -p "$(dirname "${CKPT_DST}")"

if [[ ! -f "${CKPT_SRC}" ]]; then
    echo "ColonStreamSfSNet checkpoint not found at ${CKPT_SRC}" >&2
    echo "Set COLONSTREAM_CKPT_SRC to override." >&2
    exit 1
fi

cp "${CKPT_SRC}" "${CKPT_DST}"

actual_sha256="$(sha256sum "${CKPT_DST}" | cut -d' ' -f1)"
if [[ "${actual_sha256}" != "${CKPT_SHA256}" ]]; then
    echo "Checkpoint hash mismatch:" >&2
    echo "  expected ${CKPT_SHA256}" >&2
    echo "  got      ${actual_sha256}" >&2
    echo "This is the checkpoint pinned in docs/submission-log.md;" >&2
    echo "a mismatch means either the source path moved or drifted." >&2
    exit 1
fi

echo "OK: ${CKPT_DST} (sha256 verified)"
