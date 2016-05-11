#! /bin/bash
for i in 0 1 2 3 4 5 6 7; do sudo /home/orm/IntelPerformanceCounterMonitor-PCM-V2.10/pcm-msr.x -w $1 -c $i 0x1A4 | grep core;  done
