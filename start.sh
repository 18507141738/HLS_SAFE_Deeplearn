#!/bin/bash

cd /home/xw/inst/srs/trunk
pkill srs
sleep 2
sh ingest.sh &

