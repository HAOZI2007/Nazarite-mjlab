# AMP reference motions

`go2_motion/` holds the 17 Go2 trot clips vendored from
[haozhang04/wmp_go1](https://github.com/haozhang04/wmp_go1) at `datasets/go2_motion/`.

Each file is a JSON object with `Frames` (61 scalars per frame, see
`nazarite/rl/wmp/motion.py` for the column layout), `FrameDuration` 0.02,
`LoopMode` `Wrap` and `MotionWeight` 1.0. The set covers forward / backward /
left / right at three speeds, turn-in-place at three rates, and a standing pose.

Joint order is the Go2 MJCF order (FL, FR, RL, RR) and must not be permuted.

Before publishing outside this workspace, confirm the licence of the upstream
repository: the clips were copied verbatim and no licence file was vendored with
them.
