#!/bin/bash
# ta2-vision-watchdog.sh — 60 秒检查一次，视觉查看器被动停掉就拉起来
#
# 三档语义：
#   1. /data/ta2/vision/.maintenance 存在  → 长期维护模式，永不自动拉起
#      （ta2_vision_ops.sh maintain on 创建，off 删除）
#   2. /data/ta2/vision/.stopped 存在且未超时 → 刚被"有意停止"，先别插手
#      （ta2_vision_ops.sh stop 创建；默认 600 秒后过期，防止停了忘了开）
#   3. 其它情况且不是 active → 拉起来
#
# 单元被 disable 时一律不插手。
# 手工测试过期逻辑：TA2V_STOP_GRACE=5 bash ta2-vision-watchdog.sh

set -u

UNIT=ta2-vision.service
DIR=/data/ta2/vision
MARK_MANUAL="$DIR/.maintenance"
MARK_STOP="$DIR/.stopped"
STOP_GRACE="${TA2V_STOP_GRACE:-600}"     # 有意停止的宽限期（秒）

# 1) 长期维护模式：不插手
[ -f "$MARK_MANUAL" ] && exit 0

# 2) 刚被有意停止、还在宽限期内：不插手；超时了就当作"忘了开"
if [ -f "$MARK_STOP" ]; then
  now=$(date +%s)
  mt=$(stat -c %Y "$MARK_STOP" 2>/dev/null || echo "$now")
  age=$(( now - mt ))
  if [ "$age" -lt "$STOP_GRACE" ]; then
    exit 0
  fi
  logger -t ta2-vision-watchdog "有意停止已超过 ${STOP_GRACE}s（${age}s），按遗忘处理，自动恢复"
  rm -f "$MARK_STOP"
fi

# 3) 有人明确 disable 了就别管
systemctl is-enabled --quiet "$UNIT" 2>/dev/null || exit 0

# 4) 正常跑着就什么都不做
systemctl is-active --quiet "$UNIT" && exit 0

# 5) 停了：拉起来，并在 journal 里留一句话（方便事后对时间线）
logger -t ta2-vision-watchdog "检测到 ${UNIT} 不是 active，自动拉起"
systemctl start "$UNIT" 2>/dev/null || true

sleep 2
if systemctl is-active --quiet "$UNIT"; then
  logger -t ta2-vision-watchdog "${UNIT} 已恢复"
  exit 0
fi
logger -t ta2-vision-watchdog "${UNIT} 拉起失败，看 journalctl -u ${UNIT}"
exit 1
