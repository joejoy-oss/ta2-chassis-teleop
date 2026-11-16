#!/usr/bin/env bash
# clean81 10000-step model, trained RTC, dedicated port 8006.
exec env RTC=1 /root/vla_grasp_v4_clean81_step10000.sh "$@"
