from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from humanoid_mujoco.config.g1_config import G1Config, CURRICULUM_STAGES, CURRICULUM_REWARD_THRESHOLD
from humanoid_mujoco.envs.g1_env import G1LocomotionEnv


def _run_id(log_dir: Path, reset: bool) -> int:
    """Return the PPO run ID that SB3 will use, so checkpoints match TB logs.

    SB3 increments the counter on a fresh start (reset=True) and reuses the
    latest existing ID on resume (reset=False).
    """
    max_id = 0
    for p in log_dir.glob("PPO_*"):
        if p.is_dir():
            suffix = p.name[4:]  # strip "PPO_"
            if suffix.isdigit():
                max_id = max(max_id, int(suffix))
    return max_id + 1 if reset else max(max_id, 1)


def _validate_curriculum_stage(stage: int) -> None:
    if stage < 0 or stage >= len(CURRICULUM_STAGES):
        raise ValueError(
            f"Invalid curriculum stage {stage}; expected 0..{len(CURRICULUM_STAGES) - 1}"
        )


def _final_policy_path(save_path: Path, checkpoint_dir: Path) -> Path:
    path = save_path.with_suffix("") if save_path.suffix == ".zip" else save_path
    if not path.is_absolute() and path.parent == Path("."):
        return checkpoint_dir / path.name
    return path


def make_env(config: G1Config):
    def _init():
        return G1LocomotionEnv(config)
    return _init


class CurriculumCallback(BaseCallback):
    """Advances the curriculum stage after the mean reward exceeds the threshold
    for 3 consecutive check intervals (sustained performance gate)."""

    def __init__(self, check_interval: int = 50_000, initial_stage: int = 0, verbose: int = 0) -> None:
        super().__init__(verbose)
        _validate_curriculum_stage(initial_stage)
        self.check_interval = check_interval
        self.stage = initial_stage
        self._above_count = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.check_interval != 0:
            return True
        if self.stage >= len(CURRICULUM_STAGES) - 1:
            return True

        mean_reward = self.model.ep_info_buffer
        if not mean_reward:
            return True

        avg = sum(ep["r"] for ep in mean_reward) / len(mean_reward)

        if avg > CURRICULUM_REWARD_THRESHOLD:
            self._above_count += 1
            if self.verbose:
                print(f"\n[Curriculum] Stage {self.stage} — ep_rew_mean: {avg:.2f} "
                      f"({self._above_count}/3 consecutive checks above threshold)")
            if self._above_count >= 3:
                self.stage += 1
                self._above_count = 0
                stage_cfg = CURRICULUM_STAGES[self.stage]
                if self.verbose:
                    print(f"[Curriculum] Advanced to stage {self.stage}!")
                self.training_env.env_method("update_config", **stage_cfg)
        else:
            if self._above_count > 0 and self.verbose:
                print(f"\n[Curriculum] ep_rew_mean {avg:.2f} dropped below threshold — resetting count")
            self._above_count = 0

        return True


class RewardTermLogger(BaseCallback):
    """Logs mean of each weighted reward term per rollout to TensorBoard."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._buffer: dict[str, list[float]] = defaultdict(list)

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            for name, val in info.get("reward_terms", {}).items():
                self._buffer[name].append(val)
        return True

    def _on_rollout_end(self) -> None:
        for name, vals in self._buffer.items():
            self.logger.record(f"reward_terms/{name}", float(np.mean(vals)))
        self._buffer.clear()


class VecNormalizeSaveCallback(BaseCallback):
    """Saves VecNormalize statistics alongside each model checkpoint."""

    def __init__(self, save_path: str, save_freq: int, name_prefix: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.save_freq = save_freq
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = self.save_path / f"{self.name_prefix}_{self.num_timesteps}_steps_vecnorm.pkl"
            self.training_env.save(str(path))
            if self.verbose:
                print(f"[VecNormalize] Saved stats: {path}")
        return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="G1 locomotion PPO training")
    p.add_argument("--timesteps", type=int, default=50_000_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--save-path", type=Path, default=Path("g1_policy"))
    p.add_argument("--checkpoint-freq", type=int, default=1_000_000)
    p.add_argument("--load-path", type=Path, default=None)
    p.add_argument("--curriculum-stage", type=int, default=0,
                   help="Curriculum stage to start from (for resume)")
    p.add_argument("--log-dir", type=Path, default=Path("logs"))
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _validate_curriculum_stage(args.curriculum_stage)

    config = G1Config()

    if not config.mjcf_path.exists():
        raise FileNotFoundError(
            f"MJCF not found: {config.mjcf_path}\n"
            "Clone mujoco_menagerie first:\n"
            "  git clone --depth 1 --filter=blob:none --sparse "
            "https://github.com/google-deepmind/mujoco_menagerie.git\n"
            "  git -C mujoco_menagerie sparse-checkout set unitree_g1"
        )

    n_steps = 512
    base_env = VecMonitor(
        SubprocVecEnv([make_env(G1Config()) for _ in range(args.n_envs)])
    )

    vecnorm_path = args.load_path.parent / (args.load_path.stem + "_vecnorm.pkl") if args.load_path else None

    if args.load_path is not None and vecnorm_path is not None and vecnorm_path.exists():
        env = VecNormalize.load(str(vecnorm_path), base_env)
        env.training = True
        env.norm_reward = False
        if args.verbose:
            print(f"Loaded VecNormalize stats from: {vecnorm_path}")
    else:
        env = VecNormalize(base_env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.99)

    # Apply curriculum stage (relevant for resume)
    if args.curriculum_stage > 0:
        stage_cfg = CURRICULUM_STAGES[args.curriculum_stage]
        env.env_method("update_config", **stage_cfg)
        if args.verbose:
            print(f"Applied curriculum stage {args.curriculum_stage} to all envs")

    if args.load_path is not None:
        model = PPO.load(
            str(args.load_path),
            env=env,
            tensorboard_log=str(args.log_dir),
            verbose=int(args.verbose),
        )
        if args.verbose:
            print(f"Resumed from checkpoint: {args.load_path}")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(
                net_arch=[512, 256, 128],
                activation_fn=nn.ELU,
                log_std_init=-1.0,
            ),
            learning_rate=lambda f: 3e-4 * f,
            n_steps=n_steps,
            batch_size=args.n_envs * n_steps // 4,
            n_epochs=5,
            clip_range=0.2,
            clip_range_vf=0.2,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.003,
            target_kl=0.1,
            vf_coef=1.0,
            max_grad_norm=1.0,
            tensorboard_log=str(args.log_dir),
            verbose=int(args.verbose),
        )

    run_name = f"PPO_{_run_id(args.log_dir, reset=args.load_path is None)}"
    checkpoint_dir = Path("g1_policy_checkpoints") / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.verbose:
        print(f"Checkpoints → {checkpoint_dir}/")
    save_freq = max(args.checkpoint_freq // args.n_envs, 1)

    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(checkpoint_dir),
        name_prefix=args.save_path.stem,
        verbose=int(args.verbose),
    )
    vecnorm_cb = VecNormalizeSaveCallback(
        save_path=str(checkpoint_dir),
        save_freq=save_freq,
        name_prefix=args.save_path.stem,
        verbose=int(args.verbose),
    )
    curriculum_cb = CurriculumCallback(
        check_interval=50_000,
        initial_stage=args.curriculum_stage,
        verbose=int(args.verbose),
    )
    reward_term_cb = RewardTermLogger()

    model.learn(
        total_timesteps=args.timesteps,
        callback=CallbackList([checkpoint_cb, vecnorm_cb, curriculum_cb, reward_term_cb]),
        progress_bar=True,
        reset_num_timesteps=args.load_path is None,
    )

    final_policy = _final_policy_path(args.save_path, checkpoint_dir)
    final_policy.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(final_policy))
    env.save(str(final_policy) + "_vecnorm.pkl")
    print(f"Policy saved: {final_policy}.zip")
    print(f"VecNormalize stats saved: {final_policy}_vecnorm.pkl")


if __name__ == "__main__":
    main()
