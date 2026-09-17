#!/bin/bash

# Start the triton server with the given arguments
exec tritonserver \
    --log-verbose=${WORKER_VERBOSITY:-0} \
    --model-repository=${WORKER_REPOSITORY:-/models} \
    --model-control-mode=${WORKER_CONTROL_MODE:-explicit} \
    "$@"
