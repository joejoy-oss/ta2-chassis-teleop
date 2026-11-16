#!/bin/bash
# ta2_services.sh — 五个控制台的启动/停止/状态（下载/筛选/转化/视觉/推理监控）
#   ./ta2_services.sh status
#   ./ta2_services.sh start   [all|download|review|convert|vision|monitor]
#   ./ta2_services.sh stop    [all|download|review|convert|vision|monitor]
#   ./ta2_services.sh restart [all|download|review|convert|vision|monitor]
#   ./ta2_services.sh logs    <download|review|convert|vision|monitor> [行数]

T=/data/ta2/tools
REV=/data/ta2/review
CONV=/data/ta2/converted

DL_PORT=8090
RV_PORT=8093
CV_PORT=8095
VS_PORT=8097
MN_PORT=8096

DL_PATTERN="/data/ta2/tools/ta2_download_web.py"
RV_PATTERN="/data/ta2/tools/ta2_review_server.py"
CV_PATTERN="/data/ta2/tools/ta2_convert_server.py"
VS_PATTERN="/data/ta2/tools/ta2_vision_server.py"
MN_PATTERN="/data/ta2/tools/ta2_infer_monitor.py"

port_up() { ss -ltn 2>/dev/null | grep -q ":$1 "; }

start_one() {
  case "$1" in
    download)
      port_up $DL_PORT && { echo "下载台已在 $DL_PORT"; return; }
      setsid nohup python3 "$T/ta2_download_web.py" --port $DL_PORT --root /data/ta2 \
        > /data/ta2/download_web.log 2>&1 < /dev/null &
      sleep 2; echo "下载台: http://0.0.0.0:$DL_PORT" ;;
    review)
      port_up $RV_PORT && { echo "筛选台已在 $RV_PORT"; return; }
      setsid nohup python3 "$T/ta2_review_server.py" --port $RV_PORT \
        --scan-root "$CONV" --review-root "$REV" > "$REV/server.log" 2>&1 < /dev/null &
      sleep 2; echo "筛选台: http://0.0.0.0:$RV_PORT" ;;
    convert)
      port_up $CV_PORT && { echo "转化台已在 $CV_PORT"; return; }
      setsid nohup python3 "$T/ta2_convert_server.py" --port $CV_PORT \
        --records-root /data/ta2 --convert-root "$CONV" --review-root "$REV" \
        > "$REV/convert_server.log" 2>&1 < /dev/null &
      sleep 2; echo "转化台: http://0.0.0.0:$CV_PORT" ;;
    vision)
      # 统一走 ops 脚本：它管着 systemd 和维护模式标记
      /data/ta2/tools/ta2_vision_ops.sh start ;;
    monitor)
      # 推理监控台走自己的 ops：PID 文件、配置、自检都在那边
      /data/ta2/tools/ta2_monitor_ops.sh start ;;
    *) echo "未知服务: $1" >&2; return 1 ;;
  esac
}

stop_one() {
  case "$1" in
    download) pkill -f "$DL_PATTERN" 2>/dev/null; echo "已停下载台" ;;
    review)   pkill -f "$RV_PATTERN" 2>/dev/null; echo "已停筛选台" ;;
    convert)  pkill -f "$CV_PATTERN" 2>/dev/null; echo "已停转化台" ;;
    vision)   /data/ta2/tools/ta2_vision_ops.sh stop ;;
    monitor)  /data/ta2/tools/ta2_monitor_ops.sh stop ;;
    *) echo "未知服务: $1" >&2; return 1 ;;
  esac
  sleep 1
}

status_all() {
  printf '%-10s %-6s %-8s %s\n' "服务" "端口" "状态" "入口"
  for pair in "download:$DL_PORT:$DL_PATTERN" "review:$RV_PORT:$RV_PATTERN" "convert:$CV_PORT:$CV_PATTERN" "vision:$VS_PORT:$VS_PATTERN" "monitor:$MN_PORT:$MN_PATTERN"; do
    name="${pair%%:*}"; rest="${pair#*:}"; port="${rest%%:*}"; pat="${rest#*:}"
    if port_up "$port"; then st="运行中"; else
      if pgrep -f "$pat" >/dev/null 2>&1; then st="进程在/端口不通"; else st="已停止"; fi
    fi
    printf '%-10s %-6s %-8s %s\n' "$name" "$port" "$st" "http://192.168.31.228:$port"
  done
  echo
  echo "数据集: $CONV"
  echo "标注/训练集: $REV"
}

case "${1:-status}" in
  status) status_all ;;
  start)   for s in ${2:-all}; do
             if [ "$s" = "all" ]; then for x in download review convert vision monitor; do start_one $x; done
             else start_one $s; fi
           done ;;
  stop)    for s in ${2:-all}; do
             if [ "$s" = "all" ]; then for x in download review convert vision monitor; do stop_one $x; done
             else stop_one $s; fi
           done ;;
  restart) for s in ${2:-all}; do
             if [ "$s" = "all" ]; then
               for x in download review convert vision monitor; do stop_one $x; start_one $x; done
             else stop_one $s; start_one $s; fi
           done ;;
  logs)
    case "${2:-review}" in
      download) f=/data/ta2/download_web.log ;;
      review)   f="$REV/server.log" ;;
      convert)  f="$REV/convert_server.log" ;;
      vision)   f=/data/ta2/vision/viewer.log ;;
      monitor)  f=/data/ta2/infer/monitor.log ;;
      *) echo "未知服务" >&2; exit 1 ;;
    esac
    tail -n "${3:-60}" "$f" ;;
  *) echo "用法: $0 {status|start|stop|restart|logs} [all|download|review|convert|vision|monitor]" >&2; exit 1 ;;
esac
