#!/bin/bash

workspace=$(pwd)

source ~/.bashrc


gnome-terminal -t "can1" -x sudo bash -c "cd ${workspace}/arx_can; ./arx_can0.sh; exec bash;"
