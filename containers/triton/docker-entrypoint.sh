#!/bin/bash

# Install additional Python depencencies, if the environment variable is set and not an empty string
if [ -n "$WORKER_REQUIREMENTS" ]; then
    echo "Installing additional Python dependencies:"
    echo "$WORKER_REQUIREMENTS"
    pip install --no-cache $WORKER_REQUIREMENTS
else
    echo "No additional Python dependencies to install"
fi

# Start the triton server with the given arguments
exec tritonserver \
    --log-verbose=${WORKER_VERBOSITY:-0} \
    --model-repository=${WORKER_REPOSITORY:-/models} \
    --model-control-mode=${WORKER_CONTROL_MODE:-explicit} \
    "$@"
