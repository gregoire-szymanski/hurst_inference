#!/usr/bin/env bash

max_jobs=6

for N in {10..1}; do
  python3 prediction.py --N "$N" &

  while [ "$(jobs -rp | wc -l)" -ge "$max_jobs" ]; do
    sleep 1
  done
done

wait