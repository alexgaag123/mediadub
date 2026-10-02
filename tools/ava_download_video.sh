#!/bin/bash
# 15–30 min of each AVA-AVD film, video only (h264, original resolution, 25 fps kept as source), accurate seek
v=$(grep -F -e "$1." split/video.list | head -1 | tr -d '\r'); id=$1; url="https://s3.amazonaws.com/ava-dataset/trainval/$v"
[ -s "video/$id.mp4" ] && exit 0
if ffmpeg -y -loglevel error -ss 900 -t 900 -i "$url" -an -c:v libx264 -preset veryfast -crf 20 "video/$id.tmp.mp4" 2>>video_errors.log; then
  mv "video/$id.tmp.mp4" "video/$id.mp4"; echo "$id" >> video_done.log
else echo "FAIL $id" >> video_errors.log; rm -f "video/$id.tmp.mp4"; fi
