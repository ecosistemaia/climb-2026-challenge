# CLiMB 2026: Colonoscopy Localization and Mapping Benchmark

Submission of team **MedAI** (Innovation Center for Medicine and Artificial Intelligence) to the [CLiMB 2026 Challenge](https://www.synapse.org/Synapse:syn74370700/wiki/), a monocular visual odometry task for colonoscopy video.

## Method

Depth is estimated with SfSNet (EfficientNet-B0 encoder + lightweight decoder, with a temporal Mamba block for consistency across frames), optical flow with RAFT-Small, and per-frame camera pose is recovered by lifting flow correspondences to 3D via depth and solving PnP inside RANSAC — no bundle adjustment, loop closure, or map fusion.

![The MedAI pipeline for monocular visual odometry. Dense optical-flow correspondences are lifted to 3D using predicted depth and used to estimate camera motion with PnP within RANSAC. (A) SfSNet with a temporal Mamba-2 block predicts per-frame depth while maintaining a hidden state across frames. (B) RAFT-Small estimates dense optical flow for the frame pair. (C) Kannala–Brandt rectification, EPnP + RANSAC, and clamp and chaining estimate the per-frame pose from the depth/flow correspondences. (D) The submap state machine groups these poses into trajectory segments.](imgs/pipeline3.png)

## Data

The container expects a flat directory of `.mp4` files mounted read-only at `/input`, one file per sequence (sequence name = filename without extension):

```
/input/
  Seq_001_a.mp4
  Seq_001_c.mp4
  ...
```

## Build and run

Build the image:

```bash
docker build -t climb-medai:latest climb-medai/
```

Run it against the challenge I/O contract (`/input` read-only, `/output` writable):

```bash
docker run --rm \
  --gpus all \
  --network=none \
  --memory=64g \
  --user "$(id -u)":"$(id -g)" \
  -v "${INPUT_HOST}:/input:ro" \
  -v "${OUTPUT_HOST}:/output" \
  climb-medai:latest
```

`${INPUT_HOST}` is a flat directory of `.mp4` files (see [Data](#data)); `${OUTPUT_HOST}` is where the predicted trajectories and maps are written. A local build-and-run helper is also available at `climb-medai/docker-run.sh`.

## Participants

Jean C. Polo, Diego Bravo, Diana Y. Barrero, Gabriel Pérez S., Sergio Cañar, Pablo Arbeláez, Fabio A. González, Eduardo Romero

Affiliations: Innovation Center for Medicine and Artificial Intelligence (MedAI), Universidad Nacional de Colombia, Universidad de los Andes — Bogotá, Colombia

## References

- Ruano, J., Gómez, M., Romero, E., & Manzanera, A. (2024). Leveraging a realistic synthetic database to learn shape-from-shading for estimating the colon depth in colonoscopy images. *Computerized Medical Imaging and Graphics*, 115, 102390.
- Li, H., Lu, D., Wang, J., Webster III, R. J., & Oguz, I. (2026). EndoStreamDepth: Temporally consistent monocular depth estimation for endoscopic video streams. *Proceedings of Machine Learning Research*, 315, 1697.
- Dao, T., & Gu, A. (2024). Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. *arXiv preprint arXiv:2405.21060*.

## Contact

For questions, feel free to reach out to jpoloc@unal.edu.co, dbravoh@unal.edu.co, or dbarrerom@unal.edu.co.
