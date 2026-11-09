#!/bin/bash
# $1 = label, remaining args = extra hydra overrides
LABEL=$1; shift
L=/home/kevin/duet-system1/logs_system1/T01/sonic
cd /home/kevin/third_party/GR00T-WholeBodyControl
export SONIC_DUMP_OUT=$L/native_dump_${LABEL}.npz SONIC_DUMP_ENVS=2 SONIC_DUMP_STEPS=2000
/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python /home/kevin/duet-system1/system1/sonic/native_eval_dump.py \
    +checkpoint=sonic_release/last.pt \
    +headless=True \
    ++eval_callbacks=im_eval \
    ++run_eval_loop=False \
    ++num_envs=128 \
    ++manager_env.observations.policy.enable_corruption=False \
    ++manager_env.observations.tokenizer.enable_corruption=False \
    "+manager_env/terminations=tracking/eval" \
    "++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=512" \
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered" \
    "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered" \
    "++eval_output_dir=$L/eval_${LABEL}" "$@"
