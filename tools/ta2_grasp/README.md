# 腕部视频抓取判定台

`server.py` 从视觉查看器 `http://127.0.0.1:8097/api/frame` 读取帧，在 8101
提供判定、采集、标定与参数页面。它不接收 RTP、不发布 ROS 指令，独立于机器人
编排中读取关节反馈的 `arm/grasp_monitor.py`。

运行环境共用 `/data/ta2/perception/venv`，依赖 NumPy 与 OpenCV；现场配置示例在
仓库 `deploy/config/grasp/config.json`，服务单元在 `deploy/systemd/ta2-grasp.service`。

```bash
cd /data/ta2/tools/ta2_grasp
/data/ta2/perception/venv/bin/python server.py --config /data/ta2/grasp/config.json
```

输出保存在配置指定的 `state_dir`：判定事件、采集包和帧等现场产物不入 Git。
本说明在代码交接时整理；源码采集与哈希见 `deploy/SOURCE_MANIFEST.json`，
本次未重新开展视觉判定准确率或真机抓取评估。
