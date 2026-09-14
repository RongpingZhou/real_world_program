#!/usr/bin/env bash
# Test the models trained by rwrl_067_all_games.sh, one game after another, with
# code_068_dqn_agent_test_for_real_world_input_system.py.
#
# rwrl_067_all_games.sh trains each game twice, so there are two kinds of model to test
# and code_068 loads them in two different ways:
#
#   1. code_067 writes plain state dicts to
#          saved_models/model_updates_dqn_<env id>_<steps>.pth
#      code_068 reads that list from the yml, under model_files: <env id>: code_068:,
#      and tests it with --model 2. This script writes a config listing the checkpoints
#      that are actually on disk, so freshly trained ones are picked up.
#
#   2. train.py, the rl-zoo baseline, writes a saved agent to
#          logs/<algo>/<gym id>_<run>/<gym id>.zip
#      That is an SB3 archive, not a state dict, so it cannot go in the yml list. It is
#      loaded through rl-zoo instead, with --model 1 --folder logs --exp-id 0.
#
# It runs in two sweeps, all the train.py agents first and then all the code_067
# checkpoints. A game is skipped when it has neither.
#
# code_068 appends the point estimates and confidence intervals of every checkpoint to
# one file shared by all the games, <performance dir>/<performance file>. Each row
# carries the gym id, the system the model was trained in, which code_067 recorded in a
# json next to the model, and the system this test ran in, simulation with --sensor 0
# or real_world_input with --sensor 1.
#
# Usage:
#   ./rwrl_068_all_games.sh                          # every game that has checkpoints
#   ./rwrl_068_all_games.sh Pong Gopher              # only games whose id matches these patterns
#   DRY_RUN=1 ./rwrl_068_all_games.sh                # print the commands without running them
#   DISPLAY_GAME=0 ./rwrl_068_all_games.sh           # no game window, for a headless run
#   RANDOM_POLICY=0 ./rwrl_068_all_games.sh          # no random policy baseline run
#   TEST_TRAIN_PY=0 ./rwrl_068_all_games.sh          # only the code_067 checkpoints
#   TEST_CODE_067=0 ./rwrl_068_all_games.sh          # only the train.py saved agents
#   CONFIG=real_world_system_config.yml ./rwrl_068_all_games.sh   # use an existing yml as it is
#
# Environment variables:
#   SEED           seed passed to code_068 (default 37)
#   SENSOR         --sensor, 0 tests in simulation, 1 on the real world input system
#                  (default 0)
#   PERFORMANCE_DIR  --performance-dir, where the estimates csv goes
#                  (default ../data/performance)
#   PERFORMANCE_FILE --performance-file, the csv shared by all games
#                  (default hns-estimates.csv)
#   BOOTSTRAP_REPS --bootstrap-reps behind the confidence intervals (default 50000)
#   CONFIDENCE     --confidence-interval, the coverage (default 0.95)
#   TRAINING_STEPS --training-steps, which option under training_steps in the yml
#                  code_068 reads, for example 10M, 1M, 500K (default 1M)
#   N_EPISODES     --n-episodes, how many episodes per checkpoint (default 1)
#   CUDA           --cuda value (default True)
#   MAX_EPISODE_STEPS  --max-episode-steps value (default 60000)
#   DISPLAY_GAME   --display, 1 opens the pygame window showing the game (default 1)
#   FPS            --fps, how fast the displayed game runs (default 300)
#   ZOOM           --zoom of the displayed window, 1.0 is no zoom (default 1.0)
#   PLOT           --plot, 1 shows the score plots while testing (default 0)
#   RANDOM_POLICY  --random-policy, 1 also evaluates the first file with random
#                  actions as the baseline, 0 skips that run (default 1)
#   SAVED_MODELS_DIR   where the code_067 checkpoints are (default ../data/saved_models)
#   TEST_CODE_067  1 to test the code_067 checkpoints with --model 2 (default 1)
#   TEST_TRAIN_PY  1 to test the train.py saved agents with --model 1 (default 1)
#   ZOO_LOG_DIR    --folder, where train.py wrote its runs (default logs)
#   ALGO           --algo, also the train.py sub directory (default dqn)
#   EXP_ID         --exp-id, 0 is the latest run of that game (default 0)
#   CONFIG         yml to pass to --config; unset means generate one from the checkpoints
#   BASE_CONFIG    yml the generated one starts from (default real_world_system_config.yml)
#   SCRIPT         python script to run (default code_068_..._real_world_input_system.py)
#   LOG_DIR        directory for the per game logs (default ./logs_068_all_games)
#   DRY_RUN        1 to print the code_068 commands instead of running them

set -u

SEED="${SEED:-37}"
SENSOR="${SENSOR:-0}"
PERFORMANCE_DIR="${PERFORMANCE_DIR:-../data/performance}"
PERFORMANCE_FILE="${PERFORMANCE_FILE:-hns-estimates.csv}"
BOOTSTRAP_REPS="${BOOTSTRAP_REPS:-50000}"
CONFIDENCE="${CONFIDENCE:-0.95}"
TRAINING_STEPS="${TRAINING_STEPS:-1M}"
N_EPISODES="${N_EPISODES:-1}"
CUDA="${CUDA:-True}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-60000}"
# the game window is on by default, a display is needed for it
DISPLAY_GAME="${DISPLAY_GAME:-1}"
FPS="${FPS:-300}"
ZOOM="${ZOOM:-2.0}"
PLOT="${PLOT:-0}"
RANDOM_POLICY="${RANDOM_POLICY:-1}"
SAVED_MODELS_DIR="${SAVED_MODELS_DIR:-../data/saved_models}"
TEST_CODE_067="${TEST_CODE_067:-1}"
TEST_TRAIN_PY="${TEST_TRAIN_PY:-1}"
ZOO_LOG_DIR="${ZOO_LOG_DIR:-logs}"
ALGO="${ALGO:-dqn}"
EXP_ID="${EXP_ID:-0}"
BASE_CONFIG="${BASE_CONFIG:-real_world_system_config.yml}"
SCRIPT="${SCRIPT:-code_068_dqn_agent_test_for_real_world_input_system.py}"
LOG_DIR="${LOG_DIR:-./logs_068_all_games}"
DRY_RUN="${DRY_RUN:-0}"

# the same games rwrl_067_all_games.sh trains
GAMES=(
    AlienNoFrameskip-v4
    AmidarNoFrameskip-v4
    AssaultNoFrameskip-v4
    AsterixNoFrameskip-v4
    AsteroidsNoFrameskip-v4
    AtlantisNoFrameskip-v4
    BankHeistNoFrameskip-v4
    BattleZoneNoFrameskip-v4
    BeamRiderNoFrameskip-v4
    BerzerkNoFrameskip-v4
    BowlingNoFrameskip-v4
    BoxingNoFrameskip-v4
    BreakoutNoFrameskip-v4
    CentipedeNoFrameskip-v4
    ChopperCommandNoFrameskip-v4
    CrazyClimberNoFrameskip-v4
    DefenderNoFrameskip-v4
    DemonAttackNoFrameskip-v4
    DoubleDunkNoFrameskip-v4
    EnduroNoFrameskip-v4
    FishingDerbyNoFrameskip-v4
    FreewayNoFrameskip-v4
    FrostbiteNoFrameskip-v4
    GopherNoFrameskip-v4
    GravitarNoFrameskip-v4
    HeroNoFrameskip-v4
    IceHockeyNoFrameskip-v4
    JamesbondNoFrameskip-v4
    KangarooNoFrameskip-v4
    KrullNoFrameskip-v4
    KungFuMasterNoFrameskip-v4
    MontezumaRevengeNoFrameskip-v4
    MsPacmanNoFrameskip-v4
    NameThisGameNoFrameskip-v4
    PhoenixNoFrameskip-v4
    PitfallNoFrameskip-v4
    PongNoFrameskip-v4
    PrivateEyeNoFrameskip-v4
    QbertNoFrameskip-v4
    RiverraidNoFrameskip-v4
    RoadRunnerNoFrameskip-v4
    RobotankNoFrameskip-v4
    SeaquestNoFrameskip-v4
    SkiingNoFrameskip-v4
    SolarisNoFrameskip-v4
    SpaceInvadersNoFrameskip-v4
    StarGunnerNoFrameskip-v4
    TennisNoFrameskip-v4
    TimePilotNoFrameskip-v4
    TutankhamNoFrameskip-v4
    UpNDownNoFrameskip-v4
    VentureNoFrameskip-v4
    VideoPinballNoFrameskip-v4
    WizardOfWorNoFrameskip-v4
    YarsRevengeNoFrameskip-v4
    ZaxxonNoFrameskip-v4
)

# Optional filtering: keep only games matching any of the patterns given on the command line.
if [ "$#" -gt 0 ]; then
    SELECTED=()
    for game in "${GAMES[@]}"; do
        for pattern in "$@"; do
            case "$game" in
                *"$pattern"*) SELECTED+=("$game"); break ;;
            esac
        done
    done
    if [ "${#SELECTED[@]}" -eq 0 ]; then
        echo "No game matches: $*" >&2
        exit 1
    fi
    GAMES=("${SELECTED[@]}")
fi

mkdir -p "$LOG_DIR"

# Which games have checkpoints, and the config code_068 reads them from.
# The env id mapping is the one evaluation/atari_data.py uses, so the yml keys match
# the file names the training wrote.
GENERATED_CONFIG="$LOG_DIR/generated_config_068.yml"
FOUND_FILE="$LOG_DIR/checkpoints_found.txt"

GAMES_LIST="${GAMES[*]}" \
SAVED_MODELS_DIR="$SAVED_MODELS_DIR" \
BASE_CONFIG="$BASE_CONFIG" \
GENERATED_CONFIG="$GENERATED_CONFIG" \
WRITE_CONFIG="$([ -n "${CONFIG:-}" ] && echo 0 || echo 1)" \
python - > "$FOUND_FILE" <<'PY'
import os, re, yaml
from evaluation.atari_data import get_env_id

games = os.environ["GAMES_LIST"].split()
directory = os.environ["SAVED_MODELS_DIR"]
write_config = os.environ["WRITE_CONFIG"] == "1"

per_game = {}
for gym_id in games:
    env_id = get_env_id(gym_id)
    pattern = re.compile(r"^model_updates_dqn_" + re.escape(env_id) + r"_(\d+)\.pth$")
    found = []
    for name in os.listdir(directory) if os.path.isdir(directory) else []:
        match = pattern.match(name)
        if match:
            found.append((int(match.group(1)), os.path.join(directory, name)))
    files = [path for _step, path in sorted(found)]
    if files:
        per_game[env_id] = files
    # one line per game for the shell loop: gym id, env id, how many checkpoints
    print(gym_id, env_id, len(files))

if write_config:
    with open(os.environ["BASE_CONFIG"], "r") as base:
        config = yaml.safe_load(base)
    config["model_files"] = {env_id: {"code_068": files} for env_id, files in per_game.items()}
    with open(os.environ["GENERATED_CONFIG"], "w") as generated:
        generated.write("# Generated by rwrl_068_all_games.sh from the checkpoints in "
                        + directory + ", do not edit.\n")
        yaml.safe_dump(config, generated, default_flow_style=False, sort_keys=False)
PY

status="$?"
if [ "$status" -ne 0 ]; then
    echo "Could not read the checkpoints in $SAVED_MODELS_DIR" >&2
    exit "$status"
fi

CONFIG="${CONFIG:-$GENERATED_CONFIG}"
echo "config file: $CONFIG"
if [ "$DISPLAY_GAME" = "1" ] && [ -z "${DISPLAY:-}" ]; then
    echo "warning: --display 1 but DISPLAY is not set, the game window cannot open." >&2
    echo "         run with DISPLAY_GAME=0, or xhost +local:docker on the host." >&2
fi
echo "checkpoints found: $FOUND_FILE"

TESTED=()
SKIPPED=()
FAILED=()

# Run code_068 once, with its output teed to a log file.
run_test() {
    local label="$1"
    local log="$2"
    shift 2

    echo "=============================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $label"
    echo "=============================================================="
    echo "+ $*"

    if [ "$DRY_RUN" = "1" ]; then
        TESTED+=("$label")
        return 0
    fi

    "$@" 2>&1 | tee "$log"

    local status="${PIPESTATUS[0]}"
    if [ "$status" -ne 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $label FAILED with exit code $status" >&2
        FAILED+=("$label")
        return "$status"
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $label done"
    TESTED+=("$label")
}

# what the discovery step found, one entry per game
GAME_LIST=()
ENV_ID_LIST=()
COUNT_LIST=()
while read -r game env_id count; do
    GAME_LIST+=("$game")
    ENV_ID_LIST+=("$env_id")
    COUNT_LIST+=("$count")
done < "$FOUND_FILE"

HAS_MODEL=()

# 1. every agent train.py saved, for all games, loaded through rl-zoo from its run folder
if [ "$TEST_TRAIN_PY" = "1" ]; then
    echo "=============================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] train.py agents, $ZOO_LOG_DIR/$ALGO"
    echo "=============================================================="

    for index in "${!GAME_LIST[@]}"; do
        game="${GAME_LIST[$index]}"
        env_id="${ENV_ID_LIST[$index]}"

        zoo_models=("$ZOO_LOG_DIR/$ALGO/${game}_"*/"${game}.zip")
        [ -f "${zoo_models[0]}" ] || continue
        HAS_MODEL[$index]=1

        run_test "$game train.py agent (${#zoo_models[@]} run(s) in $ZOO_LOG_DIR/$ALGO)" \
            "$LOG_DIR/${env_id}_seed${SEED}_train_py.log" \
            python "$SCRIPT" \
            --gym-id "$game" \
            --env "$game" \
            --config "$CONFIG" \
            --algo "$ALGO" \
            --folder "$ZOO_LOG_DIR" \
            --exp-id "$EXP_ID" \
            --model 1 \
            --sensor "$SENSOR" \
            --training-steps "$TRAINING_STEPS" \
            --random-policy "$RANDOM_POLICY" \
            --bootstrap-reps "$BOOTSTRAP_REPS" \
            --confidence-interval "$CONFIDENCE" \
            --performance-dir "$PERFORMANCE_DIR" \
            --performance-file "$PERFORMANCE_FILE" \
            --test 1 \
            --n-episodes "$N_EPISODES" \
            --max-episode-steps "$MAX_EPISODE_STEPS" \
            --display "$DISPLAY_GAME" \
            --fps "$FPS" \
            --zoom "$ZOOM" \
            --plot "$PLOT" \
            --cuda "$CUDA" \
            --seed "$SEED"
    done
fi

# 2. the checkpoints code_067 wrote, listed in the yml, loaded as state dicts
if [ "$TEST_CODE_067" = "1" ]; then
    echo "=============================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] code_067 checkpoints, $SAVED_MODELS_DIR"
    echo "=============================================================="

    for index in "${!GAME_LIST[@]}"; do
        game="${GAME_LIST[$index]}"
        env_id="${ENV_ID_LIST[$index]}"
        count="${COUNT_LIST[$index]}"

        [ "$count" -gt 0 ] || continue
        HAS_MODEL[$index]=1

        run_test "$game code_067 checkpoints ($env_id, $count file(s))" \
            "$LOG_DIR/${env_id}_seed${SEED}_code_067.log" \
            python "$SCRIPT" \
            --gym-id "$game" \
            --config "$CONFIG" \
            --algo "$ALGO" \
            --model 2 \
            --sensor "$SENSOR" \
            --training-steps "$TRAINING_STEPS" \
            --random-policy "$RANDOM_POLICY" \
            --bootstrap-reps "$BOOTSTRAP_REPS" \
            --confidence-interval "$CONFIDENCE" \
            --performance-dir "$PERFORMANCE_DIR" \
            --performance-file "$PERFORMANCE_FILE" \
            --test 1 \
            --n-episodes "$N_EPISODES" \
            --max-episode-steps "$MAX_EPISODE_STEPS" \
            --display "$DISPLAY_GAME" \
            --fps "$FPS" \
            --zoom "$ZOOM" \
            --plot "$PLOT" \
            --cuda "$CUDA" \
            --seed "$SEED"
    done
fi

for index in "${!GAME_LIST[@]}"; do
    [ -z "${HAS_MODEL[$index]:-}" ] && SKIPPED+=("${GAME_LIST[$index]}")
done

echo "=============================================================="
echo "ran ${#TESTED[@]} test(s):"
for label in "${TESTED[@]:-}"; do
    [ -n "$label" ] && echo "  $label"
done
echo "skipped ${#SKIPPED[@]} game(s) with no model at all"

ESTIMATES="$PERFORMANCE_DIR/$PERFORMANCE_FILE"
if [ -f "$ESTIMATES" ]; then
    echo "estimates: $ESTIMATES ($(( $(wc -l < "$ESTIMATES") - 1 )) row(s) for all games)"
else
    echo "estimates: nothing was written to $ESTIMATES"
fi
if [ "${#FAILED[@]}" -ne 0 ]; then
    echo "${#FAILED[@]} test(s) failed:"
    for label in "${FAILED[@]}"; do
        echo "  $label"
    done
    exit 1
fi
