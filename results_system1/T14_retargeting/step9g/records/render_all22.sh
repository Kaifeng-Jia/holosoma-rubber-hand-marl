#!/bin/bash
# T14 step 9g item 4 (Kevin 2026-11-26): side-by-side sheet + level grid (both verdicts) of every clip applicable at a reported
# level under either non-hand measure; videos for the clips without one (largeboxa, suitcasea: video of the same solves made by
# the driver at 02:05 / 02:10, the video carries no verdict text). First try (05:09, these two done, 3 in parallel) stopped at 05:10: 3 x 5.8 GB RSS left 5.7 GB available; rerun of the other 11 clips with 2 in parallel (+ ffmpeg), CPU.
cd /home/kevin/duet-system1
echo "git $(git rev-parse HEAD) start $(TZ=America/New_York date -Iseconds)"
/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python -B system1/t14/objects/step9g_render_both.py --logdir logs_system1/t14/step9g \
  --rejudge logs_system1/t14/rejudge_exact/step9g --fig-out results_system1/T14_retargeting/step9g/figures_all22 \
  --video-out exports_system1/T14_videos/step9g --parallel 2 \
  sub3_largebox_003 sub3_largebox_029 sub1_suitcase_031 sub1_suitcase_034 sub4_suitcase_006 sub4_suitcase_042 \
  sub1_suitcase_036 sub4_suitcase_043 sub8_suitcase_020 sub5_suitcase_046 sub1_suitcase_039
echo "rc $? end $(TZ=America/New_York date -Iseconds)"
