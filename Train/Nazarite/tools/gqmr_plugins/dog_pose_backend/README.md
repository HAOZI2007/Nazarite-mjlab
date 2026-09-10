# Nazarite GQMR dog-pose backend

This is an external `gqmr.pose_backends` plugin package.  It deliberately lives
outside GQMR so that the pose model and its heavyweight dependencies can evolve
without changing the retargeting core.

The only backend currently registered is `dog-pose-fixture`.  It emits a static
2D dog-27 skeleton for plumbing tests.  Its metadata contains
`training_eligible=false`; it is not a pose estimator and its output must never
be added to an SMP dataset.

Install it into GQMR's own environment after GQMR has been synchronized:

```bash
cd /home/haozi/桌面/GQMR-main
uv sync --frozen --extra test
uv pip install --python .venv/bin/python --no-deps -e \
  /home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/tools/gqmr_plugins/dog_pose_backend
```

The future real backend should implement the same four methods as
`FixtureDogPoseBackend`, but load SLEAP, DeepLabCut, or another dog-pose model
and return measured keypoints and confidences.
