#!/usr/bin/env bash
#
# Local build-and-test helper: build the image and run it against the sample
# videos with the same flags the scorer uses. Not submitted.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE="${IMAGE:-climb-medai:ci}"
INPUT_HOST="${INPUT_HOST:-${REPO_ROOT}/data/EndoMapper}"
OUTPUT_HOST="${OUTPUT_HOST:-${REPO_ROOT}/output}"
COLMAP_HOST="${COLMAP_HOST:-${REPO_ROOT}/data/}"
NUM_RUNS="${NUM_RUNS:-5}"
# Host GPU index to expose (container still sees it as cuda:0). GPU 0 is shared.
GPU_DEVICE="${GPU_DEVICE:-1}"

echo "== Fetch weights =="
"${SCRIPT_DIR}/scripts/fetch_weights.sh"

mkdir -p "${OUTPUT_HOST}"

# INPUT_HOST is a flat dir of *.mp4, mounted straight as /input:ro.
mapfile -t INPUT_VIDEOS < <(find "${INPUT_HOST}" -maxdepth 1 -type f -iname '*.mp4' | sort)
if [ "${#INPUT_VIDEOS[@]}" -eq 0 ]; then
  echo "No .mp4 files found under ${INPUT_HOST}" >&2
  exit 1
fi

echo "== Input: ${INPUT_HOST} (${#INPUT_VIDEOS[@]} videos) =="
echo "== Output: ${OUTPUT_HOST} =="

echo "== Build climb-medai Docker image =="
docker build -t "${IMAGE}" "${SCRIPT_DIR}"

echo "== GPU sanity check (device=${GPU_DEVICE}) =="
if docker run --rm --gpus "device=${GPU_DEVICE}" "${IMAGE}" --help >/dev/null 2>&1; then
  GPU_ARGS=(--gpus "device=${GPU_DEVICE}")
else
  echo "GPU runtime is not available; running on CPU. This will NOT meet the" >&2
  echo "runtime budget -- CPU here is for local correctness testing only." >&2
  GPU_ARGS=()
fi

echo "== Run climb-medai submission =="
docker run --rm "${GPU_ARGS[@]}" \
  --network=none \
  --memory=64g \
  --user "$(id -u)":"$(id -g)" \
  -v "${INPUT_HOST}:/input:ro" \
  -v "${OUTPUT_HOST}:/output" \
  "${IMAGE}" \
  --num-runs "${NUM_RUNS}"

echo "== Validate output tree against the submission contract =="
python3 "${REPO_ROOT}/evaluation/validate_submission.py" \
  --output "${OUTPUT_HOST}" --input "${INPUT_HOST}"

echo "== Output written to ${OUTPUT_HOST} =="
find "${OUTPUT_HOST}" -maxdepth 3 -type d | sort

echo "== Next: metrics against the COLMAP reference (${COLMAP_HOST}) =="
echo "  python evaluation/slam_evaluation.py \\"
echo "      --colmap_path ${COLMAP_HOST} \\"
echo "      --slam_path   ${OUTPUT_HOST} \\"
echo "      --results_file ${OUTPUT_HOST}/results.json"
