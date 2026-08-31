#!/usr/bin/env bash
# Generate and render the seed-43 grids for the three campaign targets.
#
# The plan asks for two seeds per target and only seed 42 was run. No training is needed: every epoch
# adapter is already on disk, so this is generation plus scoring only - 490 images at about 15 s each.
# Each task is a separate invocation so that an interruption costs one task, not all three, and each
# writes its own manifest so the images can be re-rendered later without regenerating them.
#
# Run detached; progress is in assets/seed43_{task}.log and resources in assets/seed43_resources.log.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="C:/Users/Leonardo/Desktop/zoo/dev-science-ops/unlearning/vision-unlearning"
PY="C:/Users/Leonardo/Desktop/zoo/dev-science-ops/sd-interpretability/.venv/Scripts/python.exe"
LITE="$REPO/.venv-lite/Scripts/python.exe"
MODELS="$HERE/assets/models"
export PYTHONPATH="$REPO"
export HF_HUB_DISABLE_XET=1

# CPU, memory and video memory every five minutes, so a run that starts starving is visible afterwards
# rather than only as a crash.
(
  while true; do
    "$PY" -c "
import psutil, torch, time
ram = psutil.virtual_memory()
free_vram, total_vram = torch.cuda.mem_get_info(0)
print('%s CPU %.0f%% | RAM %.1f%% used, %.1fGB free | VRAM %.2f/%.2fGB'
      % (time.strftime('%H:%M:%S'), psutil.cpu_percent(interval=1), ram.percent,
         ram.available/1024**3, (total_vram-free_vram)/1024**3, total_vram/1024**3))
" >> "$HERE/assets/seed43_resources.log" 2>&1
    sleep 300
  done
) &
MONITOR=$!
trap 'kill $MONITOR 2>/dev/null' EXIT

run_task () {
  local task="$1" model_dir="$2"
  echo "=== $(date +%H:%M:%S) starting $task"
  "$PY" "$HERE/make_epoch_grid.py" --task "$task" --seeds 43 --learning-rate 6e-4 \
      --run-suffix "_campaign_$task" --model-dir "$model_dir" \
      > "$HERE/assets/seed43_$task.log" 2>&1
  local status=$?
  echo "=== $(date +%H:%M:%S) $task finished with status $status: $(tail -1 "$HERE/assets/seed43_$task.log")"
  if [ $status -eq 0 ]; then
    "$LITE" "$HERE/make_epoch_curves.py" --task "$task" --seed 43 --run-suffix "_campaign_$task" \
        >> "$HERE/assets/seed43_$task.log" 2>&1
    echo "=== $(date +%H:%M:%S) $task curves status $?"
  fi
  return $status
}

run_task breeds "$MODELS/breeds_demo_campaign_distil_30"
run_task people "$MODELS/people_demo_campaign_people_distil_200"
run_task scenes "$MODELS/scenes_demo_campaign_scenes_distil_60"

echo "SEED43_DONE $(date +%H:%M:%S)"
