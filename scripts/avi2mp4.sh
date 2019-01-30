#!/bin/bash

for i in *.avi; do ffmpeg -i "$i" -an "${i%.*}.mp4"; done
