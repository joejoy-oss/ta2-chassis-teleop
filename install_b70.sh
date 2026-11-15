#!/usr/bin/env bash
# Install / update these scripts on B70 and restart the web remote. Run on B70 from the repo checkout.
# Robot motion is never started by this script.
set -e
cd "$(dirname "$0")"
TOOLS=/root/openpi-xpu/client/outputs/tools      # mounted as /outputs/tools in the openpi-xpu-client container
mkdir -p "$TOOLS" /root/ta2_paths
install -m 644 chassis_test.py teleop.py "$TOOLS"/
install -m 755 move.sh web_move.py ta2_client.py /root/
install -m 644 mission.py web_models.html api_docs.html vla_models.json /root/
mkdir -p /root/openpi-xpu/client/outputs/arm
install -m 644 arm/arm_pose.py arm/train_start_pose.json arm/read_arm_now.py arm/ik_offset.py arm/right_fk_fit.json arm/left_fk_fit.json arm/grasp_monitor.py /root/openpi-xpu/client/outputs/arm/
install -m 755 arm/arm.sh arm/arm_down.sh /root/
install -m 755 vla/vla_run.sh vla/vla_grasp_v3_step5000.sh vla/vla_grasp_v2_step4999.sh vla/vla_grasp_v3_step5000_rtc.sh vla/vla_grasp_v3_step10000.sh vla/vla_grasp_v3_step10000_rtc.sh vla/vla_grasp_v3_step13000.sh vla/vla_grasp_v3_step13000_rtc.sh vla/vla_grasp_v2_step4999_rtc.sh vla/vla_place.sh vla/pick_place.sh /root/
install -m 644 vla/tools/guarded_grasp.py vla/tools/test_guarded_grasp.py vla/tools/live_observation_probe.py $TOOLS/
cp -n paths/*.csv /root/ta2_paths/ 2>/dev/null || true   # keep existing paths with the same name
OLD=$(ss -ltnpH sport = :8088 | grep -o "pid=[0-9]*" | cut -d= -f2 || true)
[ -n "$OLD" ] && kill "$OLD" && sleep 1.5
nohup python3 /root/web_move.py > /root/web_move.log 2>&1 < /dev/null &
sleep 3
echo "web remote: http://$(hostname -I | awk '{print $1}'):8088"
tail -n 5 /root/web_move.log
