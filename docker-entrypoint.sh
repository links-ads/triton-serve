#!/bin/bash

start_webserver() {
    # check if the RUN_MIGRATIONS environment variable is set and is true
    if [ "$RUN_MIGRATIONS" = "true" ]; then
        echo "Running database migrations..."
        # run alembic upgrade head, exit only if it fails
        uv run alembic upgrade head || exit 1
    fi
    echo "Starting web server..."
    # if TARGET is set and equal to "prod", run the server in production mode using gunicorn
    if [ "$TARGET" = "prod" ]; then
        exec uv run gunicorn -k uvicorn.workers.UvicornWorker \
            --log-level=${LOG_LEVEL:-info} \
            --bind "0.0.0.0:5000" \
            --workers ${API_WORKERS:-4} \
            triton_serve.wsgi:app
    else
        # otherwise, run the server in development mode using uvicorn
        exec uv run uvicorn triton_serve.wsgi:app \
            --log-level=${LOG_LEVEL:-info} \
            --host 0.0.0.0 \
            --port 5000 \
            --reload
    fi
}

run_tests() {
    # check if the RUN_MIGRATIONS environment variable is set and is true
    if [ "$RUN_MIGRATIONS" = "true" ]; then
        echo "Running database migrations..."
        # run alembic upgrade head, exit only if it fails
        uv run alembic upgrade head || exit 1
    fi
    echo "Waiting for the backend to start..."

    # Set timeout (in seconds)
    timeout=60
    start_time=$(date +%s)
    while true; do
        # Try to connect to the backend
        if curl -s "http://${BACKEND_HOST}:${BACKEND_PORT}" > /dev/null; then
            echo "Backend is up!"
            break
        fi
        # Check if we've exceeded the timeout
        current_time=$(date +%s)
        if [ $((current_time - start_time)) -ge $timeout ]; then
            echo "Timeout waiting for backend to start. Exiting."
            exit 1
        fi
        echo "Backend not ready yet. Retrying in 5 seconds..."
        sleep 5
    done
    echo "Running pytest..."
    exec uv run pytest -sv --cov-report=term --log-level=${LOG_LEVEL:-info} --cov=src tests/
}

check_gpus() {
    exec nvidia-smi
}

start_sentinel() {
    echo "Starting Sentinel..."
    local worker_args=("--loglevel=${LOG_LEVEL:-info}")
    if [ "$TARGET" != "test" ]; then
        worker_args+=("--beat")
    fi
    exec uv run celery -A triton_serve.tasks worker "${worker_args[@]}"
}

main() {
    if [ "$1" = "webserver" ]; then
        start_webserver
    elif [ "$1" = "test" ]; then
        run_tests
    elif [ "$1" = "check" ]; then
        check_gpus
    elif [ "$1" = "sentinel" ]; then
        start_sentinel
    else
        echo "Invalid argument. Please specify 'webserver', 'test' or 'sentinel'."
        exit 1
    fi
}

# Call the main function passing the argument provided to the script
main "$1"
