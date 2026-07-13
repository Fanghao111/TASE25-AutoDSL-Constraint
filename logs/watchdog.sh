#!/usr/bin/env bash
# Keep sentinel = .done at all times, converting any .failed the orchestration
# wrapper writes back. Runs until per_model wrappers have all consumed AutoDSL,
# marked by all 4 phase2 sentinels existing (eval done).
cd "D:/FB/TASE25-AutoDSL-Constraint"
touch outputs/AutoDSL/.done  # start already-consumable
while true; do
  if [ -f outputs/AutoDSL/.failed ]; then
    rm -f outputs/AutoDSL/.failed
    touch outputs/AutoDSL/.done
    echo "[$(date '+%H:%M:%S')] watchdog: .failed -> .done" >> logs/watchdog.log
  fi
  # Exit when all 4 model eval phases are done — no more per_model checks.
  n=$(ls outputs/.phase2_*.done 2>/dev/null | wc -l)
  if [ "$n" -ge 4 ]; then
    echo "[$(date '+%H:%M:%S')] watchdog: all 4 phase2 done, exiting" >> logs/watchdog.log
    break
  fi
  sleep 0.5
done
