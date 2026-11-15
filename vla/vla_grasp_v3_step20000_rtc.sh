#!/usr/bin/env bash
# v3 checkpoint 20000 with async Real-Time Chunking (RTC=1 -> guarded_grasp.py --rtc). Same model, prompt, port and
# limits as /root/vla_grasp_v3_step20000.sh; only the execution changes: inference runs in the background
# while the arm keeps moving, instead of stopping every 8 steps.
exec env RTC=1 /root/vla_grasp_v3_step20000.sh "$@"
