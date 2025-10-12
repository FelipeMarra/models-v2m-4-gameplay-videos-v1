#!/bin/bash

export AUDIOCRAFT_TEAM=default
export USER=felipe
export CUDA_VISIBLE_DEVICES=0

python3 /app/code/scripts/test_suite/test_suite.py #--is_vanilla=true
#python3 /app/code/scripts/test_suite/test_suite.py --is_vanilla=false --model_sig="07ee73e6"