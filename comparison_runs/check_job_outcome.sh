#!/bin/bash
# After a job leaves the queue, check if it actually succeeded or failed.
# Usage: ./check_job_outcome.sh JOB_ID [log_prefix]
#   log_prefix e.g. per_day_study_kfold_0 (default: guess from first .err file matching JOB_ID)
# Exits 0 if all array tasks completed successfully, 1 if any failed.

JOB_ID="${1:?Usage: $0 JOB_ID [log_prefix]}"
LOG_PREFIX="${2}"
cd "$(dirname "$0")"
LOG_DIR="logs"

# Count outcomes from sacct (main array steps only, e.g. 698390_0 not .batch/.exit)
readarray -t LINES < <(sacct -j "$JOB_ID" --format=JobID,State,ExitCode -n --noheader 2>/dev/null | grep -E "^${JOB_ID}_[0-9]+[[:space:]]" || true)
COMPLETED=0
FAILED=0
CANCELLED=0
OTHER=0
for line in "${LINES[@]}"; do
  state=$(echo "$line" | awk '{print $2}')
  case "$state" in
    COMPLETED) ((COMPLETED++)) ;;
    FAILED)    ((FAILED++)) ;;
    CANCELLED|CA) ((CANCELLED++)) ;;
    *)         ((OTHER++)) ;;
  esac
done
TOTAL=$((COMPLETED + FAILED + CANCELLED + OTHER))

# If no sacct lines, job may be too old or wrong id
if [[ $TOTAL -eq 0 ]]; then
  echo "No sacct record for job $JOB_ID (old or invalid job id?)."
  exit 2
fi

echo "=============================================="
echo "Job $JOB_ID outcome summary"
echo "=============================================="
echo "  COMPLETED: $COMPLETED"
[[ $FAILED -gt 0 ]]    && echo "  FAILED:    $FAILED"
[[ $CANCELLED -gt 0 ]] && echo "  CANCELLED: $CANCELLED"
[[ $OTHER -gt 0 ]]     && echo "  Other:     $OTHER"
echo "  Total:     $TOTAL"
echo ""

if [[ $FAILED -gt 0 || $CANCELLED -gt 0 || $OTHER -gt 0 ]]; then
  echo "⚠ Not all tasks succeeded. Checking error logs..."
  if [[ -z "$LOG_PREFIX" ]]; then
    for f in "$LOG_DIR"/*_${JOB_ID}_*.err; do
      [[ -e "$f" ]] || continue
      LOG_PREFIX=$(basename "$f" .err | sed "s/_${JOB_ID}_[0-9]*$//")
      break
    done
  fi
  if [[ -n "$LOG_PREFIX" ]]; then
    SAMPLE_ERR="$LOG_DIR/${LOG_PREFIX}_${JOB_ID}_0.err"
    if [[ -f "$SAMPLE_ERR" ]]; then
      echo "First error (from ${SAMPLE_ERR}):"
      echo "----------------------------------------------"
      head -30 "$SAMPLE_ERR"
      echo "----------------------------------------------"
      echo "Fix the issue above, then resubmit. To resubmit only failed indices, use sbatch with --array and the failed task ids."
    else
      echo "No .err sample found at $SAMPLE_ERR"
    fi
  fi
  echo ""
  echo "=============================================="
  exit 1
fi

# All completed - but completed quickly often means Python crashed (exit 0 from shell)
# So peek at first .err for common failure messages
if [[ -n "$LOG_PREFIX" ]]; then
  SAMPLE_ERR="$LOG_DIR/${LOG_PREFIX}_${JOB_ID}_0.err"
else
  for f in "$LOG_DIR"/*_${JOB_ID}_*.err; do
    [[ -e "$f" ]] && SAMPLE_ERR="$f" && break
  done
fi
if [[ -f "${SAMPLE_ERR:-}" ]] && grep -q -E "Error|Exception|RuntimeError|Traceback" "$SAMPLE_ERR" 2>/dev/null; then
  echo "⚠ SLURM reported COMPLETED but stderr contains errors (e.g. Python crash):"
  echo "----------------------------------------------"
  head -25 "$SAMPLE_ERR"
  echo "----------------------------------------------"
  echo "Fix the issue above, then resubmit."
  echo "=============================================="
  exit 1
fi

echo "✅ All $TOTAL tasks completed successfully."
echo "=============================================="
exit 0
