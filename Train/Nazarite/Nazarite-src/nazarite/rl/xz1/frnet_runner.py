"""XZ1 recovery runner: persist mastery statistics alongside the FR-Net policy."""

from __future__ import annotations

from mjlab.managers.curriculum_manager import CurriculumManager
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from nazarite.mdp.xz1.frnet_curriculum import RecoveryCurriculum


class XZ1RecoveryRunner(VelocityOnPolicyRunner):
    def _recovery_curriculum(self) -> RecoveryCurriculum | None:
        manager = self.env.unwrapped.curriculum_manager
        if (
            not isinstance(manager, CurriculumManager)
            or "recovery" not in manager.active_terms
        ):
            return None
        term = manager.get_term_cfg("recovery").func
        return term if isinstance(term, RecoveryCurriculum) else None

    def save(self, path: str, infos: dict | None = None) -> None:
        curriculum = self._recovery_curriculum()
        if curriculum is not None:
            infos = {
                **(infos or {}),
                "xz1_recovery_curriculum": curriculum.state_dict(),
            }
        super().save(path, infos)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        infos = super().load(path, load_cfg, strict, map_location)
        curriculum = self._recovery_curriculum()
        if curriculum is not None:
            state = (infos or {}).get("xz1_recovery_curriculum")
            if state is not None:
                curriculum.load_state_dict(state)
            else:
                # An old high iteration counter does not establish roll mastery.
                curriculum.restart()
                print("[INFO] No recovery curriculum state; starting pose level 0.")
        return infos
