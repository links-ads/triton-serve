.ONESHELL:
PY_ENV=.venv
PY_BIN=$(shell python -c "print('$(PY_ENV)/bin') if __import__('pathlib').Path('$(PY_ENV)/bin/pip').exists() else print('')")

# Define default target
.DEFAULT_GOAL := help
DOCKER_COMPOSE := docker compose -f docker-compose.yml -f docker-compose.$(TARGET).yml

.PHONY: help
help:				## This help screen
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'


.PHONY: show
show:				## Show the current environment.
	@echo "Current environment:"
	@echo "Running using $(PY_BIN)"
	@$(PY_BIN)/python -V
	@$(PY_BIN)/python -m site


.PHONY: check-venv
check-venv:			## Check if the virtualenv exists.
	@if [ "$(PY_BIN)" = "" ]; then echo "No virtualenv detected, create one using 'make virtualenv'"; exit 1; fi


.PHONY: install
install: check-venv		## Install the project in dev mode.
	@$(PY_BIN)/pip install -e .[dev,docs,test]


.PHONY: fmt
fmt: check-venv			## Format code using black & isort.
	$(PY_BIN)/isort -v --src src/ tests/ --virtual-env $(PY_ENV)
	$(PY_BIN)/black src/ tests/


.PHONY: lint
lint: check-venv		## Run ruff, black, mypy (optional).
	@$(PY_BIN)/ruff check src/
	@$(PY_BIN)/black --check src/ tests/
	@if [ -x "$(PY_BIN)/mypy" ]; then $(PY_BIN)/mypy project_name/; else echo "mypy not installed, skipping"; fi


.PHONY: clean
clean:				## Clean unused files (VENV=true to also remove the virtualenv).
	@find ./ -name '*.pyc' -exec rm -f {} \;
	@find ./ -name '__pycache__' -exec rm -rf {} \;
	@find ./ -name 'Thumbs.db' -exec rm -f {} \;
	@find ./ -name '*~' -exec rm -f {} \;
	@rm -rf .cache
	@rm -rf .pytest_cache
	@rm -rf .mypy_cache
	@rm -rf .ruff_cache
	@rm -rf build
	@rm -rf dist
	@rm -rf *.egg-info
	@rm -rf htmlcov
	@rm -rf .tox/
	@rm -rf docs/_build
	@if [ "$(VENV)" != "" ]; then echo "Removing virtualenv..."; rm -rf $(PY_ENV); fi


.PHONY: virtualenv
virtualenv:			## Create a virtual environment.
	@echo "creating virtualenv ..."
	@if [ "$(PY_BIN)" != "" ]; then echo "virtualenv already exists, use 'make clean' to remove it."; exit; fi
	@python3 -m venv $(PY_ENV)
	@./$(PY_ENV)/bin/pip install -U pip
	@echo
	@echo "==| Please run 'source $(PY_ENV)/bin/activate' to enable the environment |=="


.PHONY: release
release:			## Create a new tag for release.
	@echo "WARNING: This operation will create s version tag and push to github"
	@read -p "Version? (provide the next x.y.z semver) : " TAG
	@VER_FILE=$$(find src -maxdepth 2 -type f -name 'version.py' | head -n 1)
	@echo "Updating version file :\n $${VER_FILE}"
	@echo __version__ = \""$${TAG}"\" > $${VER_FILE}
	@$(PY_BIN)/gitchangelog > HISTORY.md
	@git add $${VER_FILE} HISTORY.md
	@git commit -m "release: version v$${TAG} 🚀"
	@echo "creating git tag : v$${TAG}"
	@git tag v$${TAG}
	@git push -u origin HEAD --tags


check-target:				## Check if TARGET variable is set.
	@if [ -z "$(TARGET)" ]; then echo "TARGET is not set, launch the command setting TARGET=dev|prod"; exit 1; fi
	@if [ "$(TARGET)" != "dev" ] && [ "$(TARGET)" != "prod" ]; then echo "TARGET must be either dev or prod"; exit 1; fi
	@echo "Symlinking .env file to $${TARGET}.env"
	@ln -sf ./envs/$${TARGET}.env .env

.PHONY: build
build:	check-target		## Build the compose project.
	@echo "Building images with target: $${TARGET}"
	@$(DOCKER_COMPOSE) build $${ARGS}


.PHONY: up
up: check-target			## Start the project.
	@echo "Starting containers with target: $${TARGET}"
	@$(DOCKER_COMPOSE) up -d $${ARGS}

stop: check-target			## Stop the project.
	@echo "Stopping containers with target: $${TARGET}"
	@$(DOCKER_COMPOSE) stop $${ARGS}

.PHONY: down
down: check-target			## Stop the project eliminating containers, use ARGS="-v" to remove volumes.
	@echo "Stopping containers with target: $${TARGET}"
	@$(DOCKER_COMPOSE) down $${ARGS}

.PHONY: restart
restart: check-target		## Restart the project.
	@echo "Restarting containers with target: $${TARGET}"
	stop up

.PHONY: test
test:				## Run tests.
	@echo "Setting up test environment..."
	@ln -sf ./envs/test.env .env
	@echo "Executing containerized tests..."
	@export TARGET=test
	@docker compose -p serve-test \
        -f docker-compose.yml \
        -f docker-compose.test.yml up \
        --build --abort-on-container-exit --exit-code-from backend
	@echo "Tearing everything down..."
	@docker compose -p serve-test \
        -f docker-compose.yml \
        -f docker-compose.test.yml down -v

.PHONY: restart-dev
restart-dev:
	@ln -sf ./envs/dev.env .env
	@docker compose -f docker-compose.yml -f docker-compose.dev.yml down
	@docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build