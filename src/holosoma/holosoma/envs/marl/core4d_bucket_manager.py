"""Bucket-only contact reset hygiene; policy observations remain unchanged."""

from holosoma.envs.marl.core4d_smalltable_manager import Core4DSmallTableManager


class Core4DBucketManager(Core4DSmallTableManager):
    def _refresh_envs_after_reset(self, env_ids):
        super()._refresh_envs_after_reset(env_ids)
        for name in ("object_robot_contact_sensor", "object_hand_contact_sensor"):
            sensor = getattr(self.simulator, name, None)
            if sensor is not None:
                sensor.reset(env_ids)
        # Do not read freshly reset contact here: reward reads after physics.
