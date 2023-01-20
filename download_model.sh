#!/bin/bash

set -eux

sudo apt-get install qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools

if [[ -d venv ]]; then
	echo "'venv' directory found, skipping it's creation.."
else
	python3 -m venv venv
	source venv/bin/activate
	pip3 install -r requirements.txt
fi

w=weights/yolov3-tiny.weights

if [[ -f "$w" ]]; then
	echo "ok, $w found, skipping download.."
else
	wget https://github.com/smarthomefans/darknet-test/raw/master/yolov3-tiny.weights -O $w
fi

python3 tools/model_converter/convert.py cfg/yolov3-tiny.cfg weights/yolov3-tiny.weights weights/yolov3-tiny.h5
