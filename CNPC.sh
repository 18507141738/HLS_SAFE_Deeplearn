#!bin/bash

export PYTHONPATH=/home/xw/mxnet-1.5.0/python:$PYTHONPATH

cd /home/xw/PycharmProjects/workspaces/XW-demo

curdate="`date +%Y-%m-%d,%H:%M:%S`";
log_dir="log/";
echo $curdate;
echo $log_dir$curdate".log";

#python3 manager.py 2>&1 | tee $log_dir$curdate".log" &
python3 manager.py 2>&1 | tee a.log
