#!/bin/bash

start_webserver() {
    # check if the RUN_MIGRATIONS environment variable is set and is true
    if [ "$RUN_MIGRATIONS" = "true" ]; then
        echo "Running database migrations..."
        # run alembic upgrade head, exit only if it fails
        alembic upgrade head || exit 1
    fi
    echo "Starting web server..."
    # if TARGET is set and equal to "prod", run the server in production mode using gunicorn
    if [ "$TARGET" = "prod" ]; then
        exec gunicorn -k uvicorn.workers.UvicornWorker \
            --log-level=${LOG_LEVEL:-info} \
            --bind "0.0.0.0:5000" \
            --workers ${API_WORKERS:-4} \
            triton_serve.wsgi:app
    else
        # otherwise, run the server in development mode using uvicorn
        exec uvicorn triton_serve.wsgi:app \
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
        alembic upgrade head || exit 1
    fi
    echo "Running pytest..."
    exec pytest -sv --cov-report=term --log-cli-level=${LOG_LEVEL:-info} --cov=src tests/
}

check_gpus() {
    exec nvidia-smi
}

start_celery_sentinel() {
    echo "Starting Celery Sentinel..."
    if [ "$TARGET" = "test" ]; then
        exec celery -A triton_serve.tasks worker
    else    
        exec celery -A triton_serve.tasks worker -B
    fi
}

main() {
    if [ "$1" = "webserver" ]; then
        start_webserver
    elif [ "$1" = "test" ]; then
        run_tests
    elif [ "$1" = "check" ]; then
        check_gpus
    elif [ "$1" = "sentinel" ]; then
        start_celery_sentinel
    else
        echo "Invalid argument. Please specify 'webserver', 'test' or 'sentinel'."
        exit 1
    fi
}

# Call the main function passing the argument provided to the script
main "$1"
