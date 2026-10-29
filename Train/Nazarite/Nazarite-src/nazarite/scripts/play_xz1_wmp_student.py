"""Play an XZ1 visual student with its complete depth/RSSM state lifecycle."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from nazarite.rl.xz1_student import load_student_policy, student_eval_cfg
from nazarite.scripts.evaluate_xz1_wmp import TERRAINS


class StudentPlayWrapper(RslRlVecEnvWrapper):
    """Tell the visual policy about executed actions and automatic/UI resets."""

    def __init__(self, env):
        self.policy: Any = None
        super().__init__(env, clip_actions=10.0)

    def step(self, actions):
        self.policy.advance(actions)
        result = super().step(actions)
        self.policy.reset(result[2].bool())
        return result

    def reset(self):
        result = super().reset()
        if self.policy is not None:
            self.policy.reset(
                torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
            )
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--terrain", choices=TERRAINS, default="narrow_stairs")
    parser.add_argument("--difficulty", type=float, default=0.0)
    parser.add_argument("--speed", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--viewer", choices=("native", "viser"), default="native")
    parser.add_argument(
        "--command-mode", choices=("velocity", "course"), default="velocity"
    )
    parser.add_argument(
        "--vision", choices=("normal", "far", "shuffled"), default="normal"
    )
    parser.add_argument(
        "--headless-steps",
        type=int,
        default=0,
        help="Smoke test without opening a viewer",
    )
    args = parser.parse_args()
    if args.headless_steps < 0:
        parser.error("headless-steps must be nonnegative")
    cfg = student_eval_cfg(
        args.checkpoint,
        play=True,
        terrain=args.terrain,
        difficulty=args.difficulty,
        seed=args.seed,
        speed=args.speed,
        command_mode=args.command_mode,
    )
    env = StudentPlayWrapper(ManagerBasedRlEnv(cfg, device=args.device))
    try:
        policy = load_student_policy(
            env.unwrapped,
            env.get_observations(),
            args.checkpoint,
            args.device,
            args.vision,
        )
        env.policy = policy
        obs, _ = env.reset()
        if args.headless_steps:
            with torch.inference_mode():
                for _ in range(args.headless_steps):
                    obs, _, _, _ = env.step(policy(obs))
            print(f"Student play smoke passed: {args.headless_steps} steps")
        elif args.viewer == "native":
            NativeMujocoViewer(env, policy).run()
        else:
            ViserPlayViewer(env, policy).run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
