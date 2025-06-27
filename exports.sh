export CARLA_ROOT=/your-path-to-the-code?CARLA-Leaderboard-2.0/carla
export WORK_DIR=/your-path-to-the-code?CARLA-Leaderboard-2.0
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}
export SAVE_PATH=/your-path-to-the-code?CARLA-Leaderboard-2.0/results