#!/bin/bash
# Run in background: monitors current job; when it leaves the queue, submits next part (no gate on errors).
# Runs outcome check and logs result, but always submits next part even if check fails.
# Only one job in queue at a time. Loops part 0 -> 1 -> 2 -> 3 -> 4.
#
# Usage:
#   Start from scratch (submit part 0 and then chain):
#     nohup bash background_chain_kfold.sh > chain_kfold.log 2>&1 &
#   Or if part 0 is already running (e.g. job 698565), monitor it and chain from there:
#     nohup bash background_chain_kfold.sh 0 698565 >> chain_kfold.log 2>&1 &

cd "$(dirname "$0")"
INTERVAL=60

# Optional: start_part and current_job_id (if job already running)
start_part="${1:-0}"
current_job_id="$2"

if [[ -n "$current_job_id" ]]; then
  echo "$(date '+%Y-%m-%d %H:%M:%S') Monitoring existing job $current_job_id (part $start_part). When done, will submit part $((start_part+1))..."
  part="$start_part"
  job_id="$current_job_id"
else
  part=0
  echo "$(date '+%Y-%m-%d %H:%M:%S') Submitting part 0 (array 0-98)..."
  out=$(sbatch submit_per_day_study_kfold_0.slurm) || { echo "Submit failed."; exit 1; }
  job_id=$(echo "$out" | awk '{print $4}')
  echo "Part 0 submitted as job $job_id."
fi

while true; do
  # Wait until current job leaves the queue
  while squeue -j "$job_id" -h 2>/dev/null | grep -q .; do
    sleep "$INTERVAL"
  done

  echo "$(date '+%Y-%m-%d %H:%M:%S') Part $part (job $job_id) left queue. Running outcome check (log only, chain continues either way)..."
  bash check_job_outcome.sh "$job_id" "per_day_study_kfold_${part}" || true

  part=$((part + 1))
  if [[ $part -gt 4 ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') All 5 parts done."
    exit 0
  fi

  echo "$(date '+%Y-%m-%d %H:%M:%S') Submitting part $part..."
  out=$(sbatch submit_per_day_study_kfold_${part}.slurm) || { echo "Submit failed for part $part."; exit 1; }
  job_id=$(echo "$out" | awk '{print $4}')
  echo "Part $part submitted as job $job_id (array $((part*99))-$((part*99+98))). Monitoring..."
done
