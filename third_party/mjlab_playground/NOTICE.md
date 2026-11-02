The XZ1 getup port adapts code from mjlab_playground, licensed under Apache-2.0.

Copyright 2025, The mjlab Developers.

Source: https://github.com/mujocolab/mjlab_playground

Adapted files: getup/getup_env_cfg.py; getup/config/go1/env_cfgs.py and rl_cfg.py;
getup/mdp/actions.py, events.py, rewards.py and terminations.py.

Local adaptations are in Nazarite-src/nazarite/mdp/xz1/getup.py and
Nazarite-src/nazarite/config/train_config/xz1/frnet_env_cfg.py / frnet_rl_cfg.py.
They use XZ1 geometry, PD gains and joint names, safe random-pose caches,
an independent settle clock, ground-relative height, a 2 ms physics step,
equivalent self-contact cost scaling, and additional stability diagnostics.

The upstream getup task credits MuJoCo Playground (Zakka et al., 2025):
https://arxiv.org/abs/2502.08844 and https://github.com/google-deepmind/mujoco_playground/.
