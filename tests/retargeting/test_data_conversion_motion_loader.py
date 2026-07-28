from __future__ import annotations

import numpy as np

from holosoma_retargeting.data_conversion.convert_data_format_mj import MotionLoader


def test_resolve_input_fps_accepts_current_rate_metadata():
    assert MotionLoader._resolve_input_fps(np.array(30)) == 30.0
    assert MotionLoader._resolve_input_fps(np.array([50])) == 50.0


def test_resolve_input_fps_accepts_legacy_seconds_per_frame_metadata():
    np.testing.assert_allclose(
        MotionLoader._resolve_input_fps(np.array(1.0 / 30.0)),
        30.0,
    )


def test_resolve_input_fps_rejects_invalid_metadata():
    for value in (0.0, -30.0, np.nan, np.array([30.0, 30.0])):
        with np.testing.assert_raises(ValueError):
            MotionLoader._resolve_input_fps(value)
