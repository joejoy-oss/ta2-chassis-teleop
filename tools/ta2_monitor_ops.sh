#!/bin/bash
# ta2_monitor_ops.sh — 推理监控台（:8096）运维脚本
#
#   ./ta2_monitor_ops.sh start       启动（已在跑则跳过）
#   ./ta2_monitor_ops.sh stop        停止
#   ./ta2_monitor_ops.sh restart     重启
#   ./ta2_monitor_ops.sh status      摘要（curl /api/status）
#   ./ta2_monitor_ops.sh logs [n]    看日志
#   ./ta2_monitor_ops.sh selftest    合成数据自检（不占 8096）
#   ./ta2_monitor_ops.sh doctor      端口 / ffmpeg / run 根 / 进程 检查
#
# 参数覆盖：
#   PORT=8096 ./ta2_monitor_ops.sh restart
#
# 只读服务：不发布 ROS、不启停 client、不写 run 目录。
# 不要用 pkill -f ta2_ 这类宽匹配清理，会误伤视觉台/数据台。

set -u

T=/data/ta2/tools
DIR=/data/ta2/infer
SERVER="$T/ta2_infer_monitor.py"
UI="$T/infer_monitor_ui.html"
TOOL_CFG="$T/monitor.json"
CFG="${CONFIG:-$DIR/monitor.json}"
PORT="${PORT:-8096}"
PIDFILE="$DIR/monitor.pid"
LOG="$DIR/monitor.log"
PY="${PY:-/usr/bin/python3}"

mkdir -p "$DIR"

port_up() { ss -ltn 2>/dev/null | grep -q ":$PORT "; }
pid_alive() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

status_summary() {
  curl -s -m 5 "http://127.0.0.1:${PORT}/api/status" -o /tmp/ta2mon_status.json || {
    echo "拿不到 /api/status（服务没起？）"; return 1; }
  "$PY" - <<'PY'
import json
try:
    j = json.load(open('/tmp/ta2mon_status.json'))
except Exception as e:
    print('status 解析失败:', e); raise SystemExit(1)
cur = j.get('current') or {}
last = cur.get('last') or {}
win = cur.get('window') or {}
counts = cur.get('counts') or {}
print("状态      %s   版本 %s   运行 %ss" % (
    j.get('current_run') or '无 run', j.get('version'), j.get('uptime_s')))
print("run       %s  live=%s  prompt=%s  port=%s" % (
    j.get('current_run'), (cur.get('live') if cur else None),
    cur.get('prompt') or '—', cur.get('port') or '—'))
print("推理      %s 次 / actuations %s" % (
    counts.get('inferences'), counts.get('actuations')))
if last.get('budget') is not None:
    print("预算占用  %.0f%%   延迟 p50/p95 %.0f/%.0f ms" % (
        last['budget'] * 100, (win.get('sec_p50') or 0) * 1000, (win.get('sec_p95') or 0) * 1000))
if win.get('tracking_max') is not None:
    print("跟踪      max %.3f rad   关节限位/夹爪夹紧 %s / %s 次" % (
        win['tracking_max'], win.get('joint_clamps'), win.get('gripper_clamps')))
print("帧        %s" % (j.get('frames') or {}).get('enabled'))
for w in (j.get('warnings') or []):
    print("警告      %s" % w)
PY
}

doctor() {
  echo "== 监控台自检 =="
  echo "python    $("$PY" -V 2>&1)"
  echo "ffmpeg    $(command -v ffmpeg || echo '缺失（帧不可用）')"
  echo "端口      $(port_up && echo "$PORT 占用" || echo "$PORT 空闲")"
  echo "pidfile   $(pid_alive && echo "存活 $(cat "$PIDFILE")" || echo '无/已失效')"
  echo -n "run 根    "
  "$PY" - "$CFG" <<'PY'
import json, sys
try:
    print(",".join(json.load(open(sys.argv[1])).get("run_roots", [])))
except Exception:
    print("—")
PY
  for r in /root/openpi-xpu/client/outputs; do
    [ -d "$r" ] && echo "root      $r ($(ls -1 "$r" 2>/dev/null | wc -l) 项)" || echo "root      $r 不存在"
  done
  echo "磁盘      $(df -h /data | tail -1)"
  status_summary || true
}

start() {
  rm -f "$DIR/.maintenance" 2>/dev/null || true
  if port_up; then
    echo "监控台已在 $PORT 上运行"
    status_summary || true
    return 0
  fi
  if [ ! -f "$SERVER" ]; then echo "缺少 $SERVER，先部署" >&2; return 1; fi
  if [ ! -f "$UI" ]; then echo "缺少 $UI，先部署" >&2; return 1; fi
  if [ ! -f "$CFG" ]; then
    cp "$TOOL_CFG" "$CFG" 2>/dev/null || true
  fi
  if [ ! -f "$CFG" ]; then echo "缺少配置 $CFG" >&2; return 1; fi
  nohup nice -n 10 "$PY" "$SERVER" --config "$CFG" --port "$PORT" \
      >> "$LOG" 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  sleep 2
  if port_up; then
    echo "推理监控台: http://0.0.0.0:$PORT   （日志 $LOG）"
    status_summary || true
  else
    echo "启动失败，日志尾部：" >&2
    tail -n 30 "$LOG" >&2
    return 1
  fi
}

stop() {
  if pid_alive; then
    kill -TERM "$(cat "$PIDFILE")" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
      pid_alive || break
      sleep 0.4
    done
    pid_alive && kill -KILL "$(cat "$PIDFILE")" 2>/dev/null || true
  fi
  # 兜底：只匹配本服务自己的脚本名
  pkill -TERM -f "$SERVER" 2>/dev/null || true
  sleep 1
  rm -f "$PIDFILE"
  if port_up; then
    echo "仍在监听 $PORT（可能被别的进程占用）：" >&2
    ss -ltnp 2>/dev/null | grep ":$PORT " >&2
    return 1
  fi
  echo "推理监控台已停止"
}

case "${1:-status}" in
  start)    start ;;
  stop)     stop ;;
  restart)  stop; start ;;
  status)   status_summary ;;
  logs)     tail -n "${2:-80}" "$LOG" ;;
  selftest) cd /tmp && "$PY" "$SERVER" --selftest --no-frames ;;
  doctor)   doctor ;;
  *)
    echo "用法: $0 {start|stop|restart|status|logs [n]|selftest|doctor}" >&2
    exit 1 ;;
esac
