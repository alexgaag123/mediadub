#!/bin/bash
# Minutes 15-30 of each AVA-AVD film (the annotated part), stereo 44.1k FLAC audio, without downloading the whole video
v=$(echo "$1" | tr -d '\r'); id=${v%.*}; url="https://s3.amazonaws.com/ava-dataset/trainval/$v"
[ -s "audio/$id.flac" ] && exit 0
ch=$(ffprobe -v error -select_streams a:0 -show_entries stream=channels,codec_name -of csv=p=0 "$url" 2>/dev/null)
if ffmpeg -y -loglevel error -ss 900 -t 900 -i "$url" -vn -ac 2 -ar 44100 -c:a flac "audio/$id.tmp.flac" 2>>errors.log; then
  mv "audio/$id.tmp.flac" "audio/$id.flac"; echo "$id,$ch" >> channels.csv
else echo "FAIL $id" >> errors.log; rm -f "audio/$id.tmp.flac"; fi
