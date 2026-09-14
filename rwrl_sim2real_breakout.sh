#!/usr/bin/env bash
# Train and test one game on both systems, and report how far the simulation results
# carry over to the real world.
#
# The six steps are:
#
#   1. code_067 --sensor 0        trains in simulation with the 500K option of
#                                 training_steps, writes
#                                 saved_models/model_updates_dqn_<env id>_<steps>.pth
#   2. code_068 --sensor 0        tests those models in simulation, 100 episodes each
#   3. code_070 --model 2         tests the same models on the real world system
#   4. Sim2Real Gap               IQM in simulation minus IQM on the real world system,
#                                 per checkpoint, for the models of step 1
#   5. code_069                   trains the same game on the real world system, writes
#                                 code_069_model_updates_dqn_<env id>_<steps>.pth
#   6. code_070 --model 4         tests the code_069 models, 100 episodes each
#   7. Sim2Real Robustness Ratio  the IQM of step 6, trained and tested on the real
#                                 world system, divided by the IQM of step 2, trained
#                                 and tested in simulation
#
# code_068 and code_070 append their point estimates and confidence intervals to the
# same file, <performance dir>/<performance file>, shared by every game. Each row says
# which system the model was trained in and which system the test ran in, which is what
# the two numbers above are read back from.
#
# The test scripts read the checkpoints to test from the yml, so this script writes a
# config: the models of step 1 go under the code_068 option that step 2 reads and the
# code_070 option that step 3 reads, the models of step 5 go under the code_069 option
# that step 6 reads.
#
# Usage:
#   ./rwrl_sim2real_breakout.sh                      # breakout, 500K, 100 episodes
#   GAME=PongNoFrameskip-v4 ./rwrl_sim2real_breakout.sh
#   DRY_RUN=1 ./rwrl_sim2real_breakout.sh            # print the commands without running them
#   REPORT_ONLY=1 ./rwrl_sim2real_breakout.sh        # only recompute both numbers from the file
#
# Environment variables:
#   GAME           the gym id to train and test (default BreakoutNoFrameskip-v4)
#   TRAINING_STEPS --training-steps, the option under training_steps in the yml (default 500K)
#   SENSOR         --sensor for step 1 and step 2 (default 0, simulation)
#   N_EPISODES     --n-episodes for every test (default 100)
#   SEED           seed for every step (default 37)
#   ALGO           --algo for the tests (default dqn)
#   CUDA           --cuda value (default True)
#   MAX_EPISODE_STEPS  --max-episode-steps value (default 60000)
#   CHECKPOINT     --checkpoint for the trainings, 0 writes no checkpoints (default 0)
#   RANDOM_POLICY  --random-policy for the tests (default 0)
#   BOOTSTRAP_REPS --bootstrap-reps behind the confidence intervals (default 50000)
#   CONFIDENCE     --confidence-interval, the coverage (default 0.95)
#   PERFORMANCE_DIR  --performance-dir, the directory the tests write to
#                  (default ../data/performance)
#   PERFORMANCE_FILE --performance-file, the csv they share (default hns-estimates.csv)
#   GAP_FILE       the csv the gaps go to, in PERFORMANCE_DIR (default sim2real-gap.csv)
#   RATIO_FILE     the csv the ratios go to, in PERFORMANCE_DIR
#                  (default sim2real-robustness-ratio.csv)
#   DISPLAY_GAME   --display for the tests (default 1)
#   FPS ZOOM PLOT  --fps, --zoom and --plot for the tests (defaults 300, 1.0, 0)
#   CROP           --crop for code_069 and code_070, 1 crops the window (default 1)
#   SAVED_MODELS_DIR      where code_067 and code_069 write (default ../data/saved_models)
#   DATA_SAVED_MODELS_DIR the second place code_069 writes (default ../data/saved_models)
#   CHECKPOINT_DIR --checkpoint-dir for the trainings (default ./checkpoints)
#   BASE_CONFIG    the yml the generated config starts from (default real_world_system_config.yml)
#   LOG_DIR        directory for the logs (default ./logs_sim2real)
#   STEP_067 STEP_068 STEP_070 STEP_069 STEP_070_CODE_069   1 runs that step, 0 skips it
#   REPORT_ONLY    1 skips every step and only reports (default 0)
#   DRY_RUN        1 to print the commands instead of running them

set -u

GAME="${GAME:-BreakoutNoFrameskip-v4}"
TRAINING_STEPS="${TRAINING_STEPS:-500K}"
SENSOR="${SENSOR:-0}"
N_EPISODES="${N_EPISODES:-100}"
SEED="${SEED:-37}"
ALGO="${ALGO:-dqn}"
CUDA="${CUDA:-True}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-60000}"
CHECKPOINT="${CHECKPOINT:-0}"
RANDOM_POLICY="${RANDOM_POLICY:-0}"
BOOTSTRAP_REPS="${BOOTSTRAP_REPS:-50000}"
CONFIDENCE="${CONFIDENCE:-0.95}"
PERFORMANCE_DIR="${PERFORMANCE_DIR:-../data/performance}"
PERFORMANCE_FILE="${PERFORMANCE_FILE:-hns-estimates.csv}"
GAP_FILE="${GAP_FILE:-sim2real-gap.csv}"
RATIO_FILE="${RATIO_FILE:-sim2real-robustness-ratio.csv}"
DISPLAY_GAME="${DISPLAY_GAME:-1}"
FPS="${FPS:-300}"
# code_069 and code_070 run on the real system, where the window is cropped
CROP="${CROP:-1}"
ZOOM="${ZOOM:-1.0}"
PLOT="${PLOT:-0}"
SAVED_MODELS_DIR="${SAVED_MODELS_DIR:-../data/saved_models}"
DATA_SAVED_MODELS_DIR="${DATA_SAVED_MODELS_DIR:-../data/saved_models}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"
BASE_CONFIG="${BASE_CONFIG:-real_world_system_config.yml}"
LOG_DIR="${LOG_DIR:-./logs_sim2real}"
STEP_067="${STEP_067:-1}"
STEP_068="${STEP_068:-1}"
STEP_070="${STEP_070:-1}"
STEP_069="${STEP_069:-1}"
STEP_070_CODE_069="${STEP_070_CODE_069:-1}"
REPORT_ONLY="${REPORT_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"

TRAIN_067="code_067_dqn_agent_training_for_real_world_input_system.py"
SIM_TEST_068="code_068_dqn_agent_test_for_real_world_input_system.py"
TRAIN_069="code_069_dqn_agent_for_real_world_system_at_60fps.py"
REAL_TEST_070="code_070_dqn_agent_test_for_real_world_system_at_60fps.py"

ESTIMATES="$PERFORMANCE_DIR/$PERFORMANCE_FILE"
CONFIG="$LOG_DIR/config_${GAME}.yml"

if [ "$REPORT_ONLY" = "1" ]; then
    STEP_067=0; STEP_068=0; STEP_070=0; STEP_069=0; STEP_070_CODE_069=0
fi

mkdir -p "$LOG_DIR"

# Run one step, with its output teed to a log file.
run_step() {
    local label="$1"
    local log="$2"
    shift 2

    echo "=============================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $label"
    echo "=============================================================="
    echo "executing the command: $*"

    if [ "$DRY_RUN" = "1" ]; then
        return 0
    fi

    "$@" 2>&1 | tee "$log"

    local status="${PIPESTATUS[0]}"
    if [ "$status" -ne 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $label FAILED with exit code $status" >&2
        exit "$status"
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $label done"
}

# Write the config the test scripts read, from the models on disk right now.
# Prints: <env id> <code_067 model count> <code_069 model count>
write_config() {
    GAME="$GAME" \
    BASE_CONFIG="$BASE_CONFIG" \
    OUT_CONFIG="$CONFIG" \
    SAVED_MODELS_DIR="$SAVED_MODELS_DIR" \
    DATA_SAVED_MODELS_DIR="$DATA_SAVED_MODELS_DIR" \
    python - <<'PY'
import os, re, yaml
from evaluation.atari_data import get_env_id

env_id = get_env_id(os.environ["GAME"])

def find(prefix, directories):
    """the checkpoints of one game, ordered by the step in the file name"""
    pattern = re.compile(r"^" + prefix + re.escape(env_id) + r"_(\d+)\.pth$")
    found = []
    seen = set()
    # the two directories may be the same one, a file is taken once either way
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            match = pattern.match(name)
            if match:
                path = os.path.join(directory, name)
                real = os.path.realpath(path)
                if real in seen:
                    continue
                seen.add(real)
                found.append((int(match.group(1)), path))
    return [path for _step, path in sorted(found)]

saved = os.environ["SAVED_MODELS_DIR"]
data_saved = os.environ["DATA_SAVED_MODELS_DIR"]

files_067 = find(r"model_updates_dqn_", [saved])
files_069 = find(r"code_069_model_updates_dqn_", [saved, data_saved])

options = {}
if files_067:
    # code_068 tests these in simulation, code_070 tests the same ones on the rig
    options["code_068"] = files_067
    options["code_070"] = files_067
if files_069:
    options["code_069"] = files_069

with open(os.environ["BASE_CONFIG"], "r") as base:
    config = yaml.safe_load(base)
config["model_files"] = {env_id: options} if options else {}

with open(os.environ["OUT_CONFIG"], "w") as out:
    out.write("# Generated by rwrl_sim2real_breakout.sh, do not edit.\n")
    yaml.safe_dump(config, out, default_flow_style=False, sort_keys=False)

print(env_id, len(files_067), len(files_069))
PY
}

# Report one of the two numbers from the shared file.
# report <what> <first row> <output csv>
report() {
    GAME="$GAME" \
    ESTIMATES="$ESTIMATES" \
    WHAT="$1" \
    ROWS_BEFORE="$2" \
    OUT_PATH="$3" \
    python - <<'PY'
import csv, os

estimates = os.environ["ESTIMATES"]
if not os.path.exists(estimates):
    raise SystemExit(f"no results at {estimates}")

game = os.environ["GAME"]
what = os.environ["WHAT"]
first_row = int(os.environ["ROWS_BEFORE"])
out_path = os.environ["OUT_PATH"]

with open(estimates, newline="") as estimates_file:
    rows = [row for row in csv.DictReader(estimates_file)][first_row:]
rows = [row for row in rows if row["gym_id"] == game]

def latest(training_system, test_system):
    """the last row of each checkpoint trained and tested in those systems"""
    picked = {}
    for row in rows:
        if row["training_system"] == training_system and row["test_system"] == test_system:
            picked[row["model"]] = row
    return picked

def steps_of(model):
    digits = model.replace("_", "")
    return int(digits) if digits.isdigit() else -1

simulation = latest("simulation", "simulation")      # step 2, trained and tested in simulation
transferred = latest("simulation", "real_world")     # step 3, simulation model on the rig
real_world = latest("real_world", "real_world")      # step 6, trained and tested on the rig

if what == "gap":
    pairs = [(model, simulation[model], transferred[model])
             for model in simulation if model in transferred]
    if not pairs:
        print(f"no checkpoint of {game} was tested in simulation and on the real world system")
        print(f"  simulation: {sorted(simulation) or 'none'}")
        print(f"  transferred: {sorted(transferred) or 'none'}")
        raise SystemExit(0)

    header = ["datetime", "gym_id", "model", "episodes", "confidence_level",
              "iqm_simulation", "iqm_simulation_lower", "iqm_simulation_upper",
              "iqm_real_world", "iqm_real_world_lower", "iqm_real_world_upper",
              "sim2real_gap"]
    print(f"{'model':<14}{'IQM simulation':>16}{'IQM real world':>16}{'Sim2Real Gap':>16}")
else:
    pairs = [(model, simulation[model], real_world[model])
             for model in simulation if model in real_world]
    if not pairs:
        print(f"no checkpoint of {game} has both a simulation result and a real world result")
        print(f"  code_067 trained, code_068 tested: {sorted(simulation) or 'none'}")
        print(f"  code_069 trained, code_070 tested: {sorted(real_world) or 'none'}")
        raise SystemExit(0)

    header = ["datetime", "gym_id", "model", "episodes", "confidence_level",
              "iqm_simulation", "iqm_simulation_lower", "iqm_simulation_upper",
              "iqm_real_world_trained", "iqm_real_world_trained_lower",
              "iqm_real_world_trained_upper", "sim2real_robustness_ratio"]
    print(f"{'model':<14}{'IQM simulation':>16}{'IQM real world':>16}{'Robustness Ratio':>18}")

write_header = not os.path.exists(out_path)
os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

with open(out_path, "a", newline="") as out_file:
    writer = csv.writer(out_file)
    if write_header:
        writer.writerow(header)

    for model, sim_row, other_row in sorted(pairs, key=lambda pair: steps_of(pair[0])):
        sim_iqm, other_iqm = float(sim_row["iqm"]), float(other_row["iqm"])
        if what == "gap":
            value = sim_iqm - other_iqm
            print(f"{model:<14}{sim_iqm:>16.4f}{other_iqm:>16.4f}{value:>16.4f}")
        elif sim_iqm == 0:
            print(f"{model:<14}{sim_iqm:>16.4f}{other_iqm:>16.4f}{'undefined':>18}")
            continue
        else:
            # how much of the simulation result the real world run keeps
            value = other_iqm / sim_iqm
            print(f"{model:<14}{sim_iqm:>16.4f}{other_iqm:>16.4f}{value:>18.4f}")

        writer.writerow([other_row["datetime"], game, model, other_row["episodes"],
                         other_row["confidence_level"],
                         sim_iqm, sim_row["iqm_lower"], sim_row["iqm_upper"],
                         other_iqm, other_row["iqm_lower"], other_row["iqm_upper"], value])

print()
for model, sim_row, other_row in sorted(pairs, key=lambda pair: steps_of(pair[0])):
    print(f"  {model}: simulation IQM {float(sim_row['iqm']):.4f} "
          f"[{float(sim_row['iqm_lower']):.4f}, {float(sim_row['iqm_upper']):.4f}], "
          f"real world IQM {float(other_row['iqm']):.4f} "
          f"[{float(other_row['iqm_lower']):.4f}, {float(other_row['iqm_upper']):.4f}]")
print(f"\nwritten to {out_path}")
PY
}

# where the shared file ends now, so only the rows of this run are reported
ROWS_BEFORE=0
if [ -f "$ESTIMATES" ] && [ "$REPORT_ONLY" != "1" ]; then
    ROWS_BEFORE=$(( $(wc -l < "$ESTIMATES") - 1 ))
fi

# 1. train in simulation
if [ "$STEP_067" = "1" ]; then
    run_step "$GAME step 1, code_067 training in simulation (sensor $SENSOR, $TRAINING_STEPS)" \
        "$LOG_DIR/${GAME}_seed${SEED}_067_training.log" \
        python "$TRAIN_067" \
        --gym-id "$GAME" --config "$BASE_CONFIG" --training-steps "$TRAINING_STEPS" \
        --training 1 --sensor "$SENSOR" --checkpoint "$CHECKPOINT" \
        --checkpoint-dir "$CHECKPOINT_DIR" --max-episode-steps "$MAX_EPISODE_STEPS" \
        --cuda "$CUDA" --seed "$SEED"
fi

read -r ENV_ID COUNT_067 COUNT_069 <<< "$(write_config)"
echo "config file: $CONFIG"
echo "$ENV_ID: $COUNT_067 code_067 model(s), $COUNT_069 code_069 model(s)"

# 2. test them in simulation
if [ "$STEP_068" = "1" ]; then
    run_step "$GAME step 2, code_068 test in simulation ($COUNT_067 checkpoint(s), $N_EPISODES episodes)" \
        "$LOG_DIR/${GAME}_seed${SEED}_068_test.log" \
        python "$SIM_TEST_068" \
        --gym-id "$GAME" --config "$CONFIG" --algo "$ALGO" --model 2 --sensor "$SENSOR" \
        --test 1 --training-steps "$TRAINING_STEPS" --n-episodes "$N_EPISODES" \
        --random-policy "$RANDOM_POLICY" --bootstrap-reps "$BOOTSTRAP_REPS" \
        --confidence-interval "$CONFIDENCE" --performance-dir "$PERFORMANCE_DIR" \
        --performance-file "$PERFORMANCE_FILE" --max-episode-steps "$MAX_EPISODE_STEPS" \
        --display "$DISPLAY_GAME" --fps "$FPS" --zoom "$ZOOM" --plot "$PLOT" \
        --cuda "$CUDA" --seed "$SEED"
fi

# 3. test the same models on the real world system
if [ "$STEP_070" = "1" ]; then
    run_step "$GAME step 3, code_070 test of the simulation models ($COUNT_067 checkpoint(s), $N_EPISODES episodes)" \
        "$LOG_DIR/${GAME}_seed${SEED}_070_test_simulation_models.log" \
        python "$REAL_TEST_070" \
        --gym-id "$GAME" --env "$GAME" --config "$CONFIG" --algo "$ALGO" --model 2 \
        --test 1 --training-steps "$TRAINING_STEPS" --n-episodes "$N_EPISODES" \
        --random-policy "$RANDOM_POLICY" --bootstrap-reps "$BOOTSTRAP_REPS" \
        --confidence-interval "$CONFIDENCE" --performance-dir "$PERFORMANCE_DIR" \
        --performance-file "$PERFORMANCE_FILE" --max-episode-steps "$MAX_EPISODE_STEPS" \
        --display "$DISPLAY_GAME" --fps "$FPS" --zoom "$ZOOM" --plot "$PLOT" \
        --crop "$CROP" --cuda "$CUDA" --seed "$SEED"
fi

# 4. the Sim2Real Gap
echo "=============================================================="
echo "Sim2Real Gap, IQM in simulation minus IQM on the real world system"
echo "=============================================================="
if [ "$DRY_RUN" = "1" ]; then
    echo "dry run, nothing was tested"
else
    report gap "$ROWS_BEFORE" "$PERFORMANCE_DIR/$GAP_FILE"
fi

# 5. train on the real world system
if [ "$STEP_069" = "1" ]; then
    run_step "$GAME step 5, code_069 training on the real world system ($TRAINING_STEPS)" \
        "$LOG_DIR/${GAME}_seed${SEED}_069_training.log" \
        python "$TRAIN_069" \
        --gym-id "$GAME" --config "$CONFIG" --training-steps "$TRAINING_STEPS" \
        --training 1 --checkpoint "$CHECKPOINT" --checkpoint-dir "$CHECKPOINT_DIR" \
        --max-episode-steps "$MAX_EPISODE_STEPS" --cuda "$CUDA" --fps "$FPS" \
        --crop "$CROP" --seed "$SEED"

    # the models of step 5 have to be in the config before step 6 reads it
    read -r ENV_ID COUNT_067 COUNT_069 <<< "$(write_config)"
    echo "$ENV_ID: $COUNT_067 code_067 model(s), $COUNT_069 code_069 model(s)"
fi

# 6. test the code_069 models on the real world system
if [ "$STEP_070_CODE_069" = "1" ]; then
    if [ "$COUNT_069" -eq 0 ] && [ "$DRY_RUN" != "1" ]; then
        echo "No code_069 model to test, step 5 wrote nothing" >&2
        exit 1
    fi

    run_step "$GAME step 6, code_070 test of the code_069 models ($COUNT_069 checkpoint(s), $N_EPISODES episodes)" \
        "$LOG_DIR/${GAME}_seed${SEED}_070_test_code_069_models.log" \
        python "$REAL_TEST_070" \
        --gym-id "$GAME" --env "$GAME" --config "$CONFIG" --algo "$ALGO" --model 4 \
        --test 1 --training-steps "$TRAINING_STEPS" --n-episodes "$N_EPISODES" \
        --random-policy "$RANDOM_POLICY" --bootstrap-reps "$BOOTSTRAP_REPS" \
        --confidence-interval "$CONFIDENCE" --performance-dir "$PERFORMANCE_DIR" \
        --performance-file "$PERFORMANCE_FILE" --max-episode-steps "$MAX_EPISODE_STEPS" \
        --display "$DISPLAY_GAME" --fps "$FPS" --zoom "$ZOOM" --plot "$PLOT" \
        --crop "$CROP" --cuda "$CUDA" --seed "$SEED"
fi

# 7. the Sim2Real Robustness Ratio
echo "=============================================================="
echo "Sim2Real Robustness Ratio, IQM of code_069 trained and code_070 tested"
echo "divided by IQM of code_067 trained and code_068 tested"
echo "=============================================================="
if [ "$DRY_RUN" = "1" ]; then
    echo "dry run, nothing was tested"
    exit 0
fi
report ratio "$ROWS_BEFORE" "$PERFORMANCE_DIR/$RATIO_FILE"
