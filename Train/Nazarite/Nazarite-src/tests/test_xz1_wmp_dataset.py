"""Temporal alignment, split isolation and real offline model update coverage."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from nazarite.config.train_config.XZ1_env_cfgs.collect_env_cfg import XZ1DepthCfg
from nazarite.config.train_config.XZ1_env_cfgs.teacher_env_cfg import (
    Nazarite_WMP_HIM_Eval_XZ1,
)
from nazarite.rl.wmp.world_model import WMPWorldModel, WMPWorldModelCfg
from nazarite.rl.wmp.xz1_dataset import (
    EpisodeWriter,
    XZ1SequenceDataset,
    sanitize_xz1_depth,
    validate_episode,
    xz1_wm_prop,
)
from nazarite.scripts.collect_xz1_wmp import collection_cfg
from nazarite.scripts.pretrain_xz1_wmp import run_pretraining, validate_world_model


def frames(seed, count=8):
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for i in range(count):
        rows.append(
            {
                "depth_m": torch.rand(64, 64, 1, generator=generator).half() * 2,
                "prop": torch.rand(33, generator=generator),
                "actor_history": torch.zeros(6, 47),
                "teacher_scan": torch.zeros(132),
                "teacher_latent": torch.rand(32, generator=generator),
                "teacher_action": torch.full((12,), float(i + 1)),
                "action": torch.full((60,), float(i)),
                "is_first": torch.tensor(i == 0),
                "time_s": torch.tensor(i * 0.1, dtype=torch.float64),
                "global_time_s": torch.tensor(i * 0.1, dtype=torch.float64),
            }
        )
    return rows


def dataset(tmp_path):
    directory = tmp_path / "data"
    writer = EpisodeWriter(directory, {"camera": asdict(XZ1DepthCfg())})
    for seed, split in ((42, "train"), (43, "validation")):
        writer.add(frames(seed), {"seed": seed, "split": split})
    writer.finish()
    return directory


def test_xz1_prop_drops_phase_actions_and_true_velocity():
    estimator = torch.arange(50).float().reshape(1, 50)
    expected = torch.cat((estimator[:, :9], estimator[:, 11:35]), -1)
    torch.testing.assert_close(xz1_wm_prop(estimator), expected)
    assert xz1_wm_prop(estimator).shape == (1, 33)
    with pytest.raises(ValueError, match="50-D"):
        xz1_wm_prop(estimator[:, :48])


def test_warp_depth_misses_are_far_instead_of_near_obstacles():
    depth = torch.tensor([0.0, float("nan"), float("inf"), -1.0, 0.25, 3.0])
    torch.testing.assert_close(
        sanitize_xz1_depth(depth), torch.tensor([2.0, 2.0, 2.0, 2.0, 0.25, 2.0])
    )


def test_sample_windows_reset_latent_and_never_mix_splits(tmp_path):
    directory = dataset(tmp_path)
    train = XZ1SequenceDataset(directory, "train", 4)
    validation = XZ1SequenceDataset(directory, "validation", 4)
    assert len(train) == len(validation) == 5
    window = train[3]
    assert window["image"].dtype == torch.uint8
    assert window["image"].shape == (4, 64, 64, 1)
    assert window["is_first"].tolist() == [True, False, False, False]
    assert window["action"][0].count_nonzero() == 0
    assert window["action"][1, 0] == 4
    assert not torch.equal(train[0]["image"], validation[0]["image"])
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["episodes"][1]["seed"] = 42
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="leakage"):
        XZ1SequenceDataset(directory, "train", 4)


def test_rejects_cross_episode_or_wrong_time_data():
    data = {key: torch.stack([row[key] for row in frames(1)]) for key in frames(1)[0]}
    validate_episode(data, 2.0)
    data["time_s"][3] += 0.02
    with pytest.raises(ValueError, match="10 Hz"):
        validate_episode(data, 2.0)
    data["time_s"][3] -= 0.02
    data["is_first"][3] = True
    with pytest.raises(ValueError, match="one initial"):
        validate_episode(data, 2.0)


def test_eval_jitter_and_privileged_command_modes_are_explicit():
    fixed = Nazarite_WMP_HIM_Eval_XZ1(reset_yaw=0.08)
    course = Nazarite_WMP_HIM_Eval_XZ1(command_mode="course", speed=0.6)
    assert fixed.events["reset_base"].params["pose_range"]["yaw"] == (-0.08, 0.08)
    assert type(fixed.commands["twist"]).__name__ == "UniformVelocityCommandCfg"
    assert course.commands["twist"].narrow_speed_range == (0.6, 0.6)
    assert not fixed.auto_reset and not course.auto_reset


def test_collection_filters_command_type_names_to_the_fixed_case():
    args = SimpleNamespace(
        camera_pos=(0.2, 0.0, 0.08),
        camera_fovy=58.0,
        nominal=False,
        num_envs=2,
        episodes_per_case=3,
        episode_seconds=0.14,
    )
    cfg, _ = collection_cfg(args, "narrow_stairs", 0.5, 44)
    assert cfg.commands["twist"].forward_only_terrain_names == ()
    assert cfg.commands["twist"].narrow_speed_range == (0.3, 0.7)
    assert set(cfg.scene.terrain.terrain_generator.sub_terrains) == {"narrow_stairs"}
    assert not cfg.auto_reset


def test_offline_update_validation_checkpoint_and_resume(tmp_path):
    directory = dataset(tmp_path)
    cfg = WMPWorldModelCfg(
        action_chunk=5,
        cnn_depth=2,
        prop_hidden_dim=8,
        prop_layers=1,
        deter_dim=8,
        stoch_groups=2,
        stoch_classes=2,
        rssm_hidden_dim=8,
    )
    args = SimpleNamespace(
        dataset=directory,
        output=tmp_path / "wm",
        resume=None,
        updates=2,
        batch_size=2,
        batch_length=4,
        eval_interval=2,
        validation_batches=1,
        seed=42,
        device="cpu",
    )
    history = run_pretraining(args, cfg)
    assert history[-1]["step"] == 2
    for row in history:
        assert "future_to_persistence_ratio" in row["validation"]
        assert "terrain_probe_mse" in row["validation"]
        assert all(torch.isfinite(torch.tensor(v)) for v in row["validation"].values())
    checkpoint = torch.load(args.output / "wm_2.pt", weights_only=True)
    assert checkpoint["world_model_implementation"] == "wmp_go1_dreamer_v1"
    assert checkpoint["optimizer"]["state"]
    assert checkpoint["model_cfg"]["deter_dim"] == 8
    args.resume = args.output / "wm_2.pt"
    args.output = tmp_path / "wm_resume"
    args.updates = 3
    history = run_pretraining(args, cfg)
    assert history[-1]["step"] == 3
    # Old compact-model checkpoints must fail explicitly, even when --resume
    # points to a syntactically valid file with the same data manifest.
    checkpoint.pop("world_model_implementation")
    legacy = tmp_path / "compact.pt"
    torch.save(checkpoint, legacy)
    args.resume = legacy
    args.output = tmp_path / "must_not_be_created"
    with pytest.raises(ValueError, match="compact World Model is incompatible"):
        run_pretraining(args, cfg)
    assert not args.output.exists()


def test_reference_validation_is_repeatable_and_preserves_training_rng(tmp_path):
    directory = dataset(tmp_path)
    train = XZ1SequenceDataset(directory, "train", 4)
    validation = XZ1SequenceDataset(directory, "validation", 4)
    model = WMPWorldModel(
        WMPWorldModelCfg(
            cnn_depth=2,
            prop_hidden_dim=8,
            prop_layers=1,
            deter_dim=8,
            stoch_groups=2,
            stoch_classes=2,
            rssm_hidden_dim=8,
            head_units=8,
            action_chunk=5,
        )
    )
    rng = torch.random.get_rng_state().clone()
    first = validate_world_model(model, train, validation, batch_size=2, batches=1)
    torch.testing.assert_close(torch.random.get_rng_state(), rng)
    second = validate_world_model(model, train, validation, batch_size=2, batches=1)
    assert first == second
    assert model.training
