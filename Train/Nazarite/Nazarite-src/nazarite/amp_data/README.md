# AMP reference motions

`go2_motion/` holds the 17 Go2 trot clips vendored from
[haozhang04/wmp_go1](https://github.com/haozhang04/wmp_go1) at `datasets/go2_motion/`.

Each file is a JSON object with `Frames` (61 scalars per frame, see
`nazarite/rl/wmp/motion.py` for the column layout), `FrameDuration` 0.02,
`LoopMode` `Wrap` and `MotionWeight` 1.0. Coverage: forward and backward at three
speeds each (0.4 / 0.8 / 1.3 and 0.4 / 0.7 / 1.0 m/s), sideways at two speeds
each (0.2 / 0.5 m/s), turn-in-place at three rates (0.2 / 0.5 / 0.8 rad/s), and
one standing pose.

Joint order is the Go2 MJCF order (FL, FR, RL, RR) and must not be permuted.

The upstream repository is Apache-2.0; the clips are copied verbatim, so a
distribution of this repository has to carry that licence and attribution
alongside `go2_motion/`.
