#!/bin/bash
cd /workspace/frc-cad
nohup python3 full_run.py >> full/stager.out 2>&1 &
echo $! > full/stager.pid
