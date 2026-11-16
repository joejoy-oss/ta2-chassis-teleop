#!/bin/bash
# ta2_vision_ops.sh — 视觉查看器（:8097）运维脚本
#
#   ./ta2_vision_ops.sh start          启动（已在跑则跳过）
#   ./ta2_vision_ops.sh stop           停止（含 ffmpeg 子进程）
#   ./ta2_vision_ops.sh restart        重启（改完参数用这个）
#   ./ta2_vision_ops.sh status         打印 /api/status 摘要
#   ./ta2_vision_ops.sh logs [行数]    看日志
#   ./ta2_vision_ops.sh selftest       不开服务，等 30 帧自检解码链路
#   ./ta2_vision_ops.sh maintain on    进维护模式（暂停看门狗自动拉起）
#   ./ta2_vision_ops.sh maintain off   退出维护模式
#
# 装了 systemd 单元（ta2-vision.service）就走 systemctl：
#   * 崩了 3 秒后自动拉起
#   * B70 重启后自动起来
#   * systemctl stop 才会真停
# 没装单元则退回 setsid nohup 的裸进程方式。
#
# 参数覆盖（会写进 /data/ta2/vision/env）：
#   PORT=8097 RTP_PORT=8890 WIDTH=640 FPS=12 QUALITY=6 ./ta2_vision_ops.sh restart

set -u

T=/data/ta2/tools
LOG_DIR=/data/ta2/vision
LOG="$LOG_DIR/viewer.log"
ENV_FILE="$LOG_DIR/env"
SERVER="$T/ta2_vision_server.py"
REGIONS="$T/regions.json"
UNIT=ta2-vision.service

# 记录调用方是否显式指定了参数：显式指定才覆盖 systemd 的 EnvironmentFile，
# 否则保留网页「高清/标准」按钮上次选的分辨率。
EXPLICIT=0
for _v in PORT RTP_PORT WIDTH FPS QUALITY; do
  eval "[ -n \"\${$_v+x}\" ]" && EXPLICIT=1
done

PORT="${PORT:-8097}"
RTP_PORT="${RTP_PORT:-8890}"
WIDTH="${WIDTH:-640}"
FPS="${FPS:-12}"
QUALITY="${QUALITY:-6}"

SDP_MARK="ta2_vision_${RTP_PORT}.sdp"
# 标记语义：
#   .maintenance —— maintain on 创建，长期维护模式，看门狗永不自动拉起
#   .stopped     —— stop 创建，有意停止；看门狗宽限 10 分钟后按"忘了开"处理并恢复
MARK="$LOG_DIR/.maintenance"
MARK_STOP="$LOG_DIR/.stopped"
STOP_GRACE="${TA2V_STOP_GRACE:-600}"

port_up() { ss -ltn 2>/dev/null | grep -q ":${PORT} " ; }
have_unit() { [ -f "/etc/systemd/system/$UNIT" ] ; }

write_env() {
  mkdir -p "$LOG_DIR"
  {
    echo "TA2V_PORT=${PORT}"
    echo "TA2V_RTP_PORT=${RTP_PORT}"
    echo "TA2V_WIDTH=${WIDTH}"
    echo "TA2V_FPS=${FPS}"
    echo "TA2V_QUALITY=${QUALITY}"
    echo "TA2V_REGIONS=${REGIONS}"
    echo "TA2V_LOG_DIR=${LOG_DIR}"
  } > "$ENV_FILE"
}

status_summary() {
  curl -s -m 5 "http://127.0.0.1:${PORT}/api/status" -o /tmp/ta2v_status.json || {
    echo "拿不到 /api/status（服务没起？）"; return 1; }
  python3 - <<'PY'
import json
try:
    j = json.load(open('/tmp/ta2v_status.json'))
except Exception as e:
    print('status 解析失败:', e); raise SystemExit(1)
f = j.get('frame') or {}
loss = j.get('loss') or {}
print("状态      %s   后端 %s   pid %s" % (j.get('state'), j.get('backend'), j.get('pid')))
print("画面      %sx%s  %s 字节  age=%sms" % (f.get('w'), f.get('h'), f.get('bytes'), f.get('age_ms')))
print("帧率      %s fps   累计 %s 帧   在线 %s" % (j.get('fps'), j.get('frames_total'), j.get('viewers')))
print("丢包      missed=%s jitter_full=%s decode_errors=%s" %
      (loss.get('missed'), loss.get('jitter_full'), loss.get('decode_errors')))
print("重启      %s 次   运行 %ss" % (j.get('restarts'), j.get('uptime_s')))
if j.get('note'):
    print("备注      %s" % j['note'])
tail = j.get('err_tail') or []
if tail:
    print("ffmpeg    " + tail[-1][:150])
if j.get('state') != 'live':
    raise SystemExit(3)
PY
}

start() {
  mkdir -p "$LOG_DIR"
  rm -f "$MARK" "$MARK_STOP"
  if have_unit; then
    [ "$EXPLICIT" = "1" ] && write_env
    systemctl start "$UNIT"
    sleep 3
    if port_up; then
      echo "视觉查看器: http://0.0.0.0:${PORT}   （systemd: $UNIT，日志 $LOG）"
      status_summary || true
    else
      echo "启动失败，最近日志：" >&2
      systemctl --no-pager -l status "$UNIT" | tail -n 15 >&2
      tail -n 20 "$LOG" >&2
      return 1
    fi
    return 0
  fi

  # ---- 没有 systemd 单元时的兜底路径 ----
  if port_up; then
    echo "视觉查看器已在 ${PORT} 上运行"
    status_summary || true
    return 0
  fi
  if [ ! -f "$SERVER" ]; then
    echo "缺少 $SERVER，先部署" >&2; return 1
  fi
  mkdir -p "$LOG_DIR"
  pkill -f "$SDP_MARK" 2>/dev/null
  setsid nohup python3 "$SERVER" \
      --port "$PORT" --rtp-port "$RTP_PORT" \
      --width "$WIDTH" --fps "$FPS" --quality "$QUALITY" \
      --regions "$REGIONS" --log-dir "$LOG_DIR" \
      >> "$LOG" 2>&1 < /dev/null &
  sleep 3
  if port_up; then
    echo "视觉查看器: http://0.0.0.0:${PORT}   （裸进程，日志 $LOG）"
    status_summary || true
  else
    echo "启动失败，日志尾部：" >&2
    tail -n 20 "$LOG" >&2
    return 1
  fi
}

stop() {
  mkdir -p "$LOG_DIR"
  touch "$MARK_STOP"
  echo "⚠️  停掉之后，手机/电脑上的 http://<B70>:8097 就打不开了。" >&2
  echo "    只是想腾出 UDP 8890 给 Windows 的 ta2-vision？看完执行 ta2_vision_ops.sh start。" >&2
  echo "    （有意停止标记：$MARK_STOP，${STOP_GRACE}s 后看门狗会按遗忘处理自动恢复）" >&2
  echo "    要停更久：先 ta2_vision_ops.sh maintain on" >&2
  if have_unit; then
    systemctl stop "$UNIT"
  fi
  pkill -TERM -f "$SERVER" 2>/dev/null
  sleep 2
  pkill -KILL -f "$SERVER" 2>/dev/null
  pkill -KILL -f "$SDP_MARK" 2>/dev/null
  sleep 1
  if port_up; then
    echo "仍在监听 ${PORT}：" >&2
    ss -ltnp 2>/dev/null | grep ":${PORT} " >&2
    return 1
  fi
  echo "视觉查看器已停止"
}

case "${1:-status}" in
  start)    start ;;
  stop)     stop ;;
  restart)  stop; start ;;
  status)
    if have_unit; then
      systemctl is-active --quiet "$UNIT" && echo "systemd: active（$(systemctl is-enabled "$UNIT" 2>/dev/null)）" \
        || echo "systemd: $(systemctl is-active "$UNIT" 2>/dev/null)"
    else
      echo "systemd: 未安装单元（裸进程模式）"
    fi
    status_summary ;;
  logs)     tail -n "${2:-60}" "$LOG" ;;
  maintain)
    mkdir -p "$LOG_DIR"
    case "${2:-on}" in
      on)  touch "$MARK"; rm -f "$MARK_STOP"
           echo "维护模式 ON：看门狗暂停自动拉起（$MARK，长期有效）" ;;
      off) rm -f "$MARK"; echo "维护模式 OFF：看门狗恢复值守" ;;
      *) echo "用法: $0 maintain on|off" >&2; exit 1 ;;
    esac ;;
  selftest)
    # 自检要独占 UDP 8890，先把常驻服务停掉，跑完再拉起来
    was_running=0
    if port_up; then was_running=1; stop; fi
    python3 "$SERVER" --selftest --selftest-frames 30 \
        --rtp-port "$RTP_PORT" --width "$WIDTH" --fps "$FPS" \
        --quality "$QUALITY" --regions "$REGIONS" --log-dir "$LOG_DIR"
    rc=$?
    if [ "$was_running" = "1" ]; then
      echo "--- 自检结束，恢复常驻服务 ---"
      start
    fi
    exit $rc ;;
  *)
    echo "用法: $0 {start|stop|restart|status|logs|selftest|maintain on|off}" >&2
    exit 1 ;;
esac
