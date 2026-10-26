#!/bin/bash
# soak.sh —— 长稳记录（默认 20 次 × 30 s = 10 分钟）
# 用法：bash scripts/soak.sh [次数]
# 前提：另开窗口先跑 run_demo.sh --voice --tts --push；
#       本脚本第一行若显示 demo_rss=NA，说明 demo 没在跑，这轮数据无效。

N="${1:-20}"
for i in $(seq 1 "$N"); do
  t=$(date +%H:%M:%S)

  z=""
  for f in /sys/class/thermal/thermal_zone*/temp; do
    z="$z $(( $(cat "$f") / 1000 ))C"
  done

  # NR==2 = 物理内存那行，避开中文"内存："导致 /Mem:/ 匹配失败
  mem=$(free -m | awk 'NR==2{printf "used=%sMB avail=%sMB", $3, $7}')

  pid=$(pgrep -f "build/demo" | head -1)
  if [ -n "$pid" ]; then
    rss=$(awk '/VmRSS/{printf "%dMB", $2/1024}' /proc/"$pid"/status)
    thr=$(awk '/Threads/{print $2}' /proc/"$pid"/status)
    demo="demo_rss=$rss thr=$thr"
  else
    demo="demo_rss=NA(未运行!)"
  fi

  load=$(cut -d" " -f1-3 /proc/loadavg)
  f4=$(cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq 2>/dev/null)

  echo "$t zones:$z $mem $demo load=$load cpu4=${f4}kHz"
  sleep 30
done