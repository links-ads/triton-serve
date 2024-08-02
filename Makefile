.ONESHELL:
PY_ENV=.venv
PY_BIN=$(shell python -c "print('$(PY_ENV)/bin') if __import__('pathlib').Path('$(PY_ENV)/bin/pip').exists() else print('')")

# Define default variables
.DEFAULT_GOAL := help
PROFILE := gpu
DOCKER_COMPOSE := docker compose --profile $(PROFILE) -f docker-compose.yml -f docker-compose.$(TARGET).yml

.PHONY: help
help:				## This help screen
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'


.PHONY: check-venv
check-venv:			## Check if the virtualenv exists.
	@if [ "$(PY_BIN)" = "" ]; then echo "No virtualenv detected, create one using 'make virtualenv'"; exit 1; fi


.PHONY: install
install: check-venv		## Install the project in dev mode.
	@$(PY_BIN)/pip install -e .[dev,docs,test]


.PHONY: fmt
fmt: check-venv			## Format code using ruff.
	@$(PY_BIN)/ruff format --check .



.PHONY: lint
lint: check-venv		## Run ruff, mypy (optional).
	@$(PY_BIN)/ruff check .
	@$(PY_BIN)/ruff format --check .
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
	@git add .
	@git commit -m "release: version v$${TAG} 🚀"
	@echo "creating git tag : v$${TAG}"
	@git tag v$${TAG}
	@git push -u origin HEAD --tags


check-target:				## Check if TARGET variable is set.
	@if [ -z "$(TARGET)" ]; then echo "TARGET is not set, launch the command setting TARGET=dev|prod"; exit 1; fi
	@if [ "$(TARGET)" != "dev" ] && [ "$(TARGET)" != "prod" ]; then echo "TARGET must be either 'dev' or 'prod'"; exit 1; fi
	@echo "Symlinking .env file to $${TARGET}.env"
	@if [ ! -f ./envs/$${TARGET}.env ]; then echo "File ./envs/$${TARGET}.env does not exist, please create it"; exit 1; fi
	@ln -sf ./envs/$${TARGET}.env .env


check-profile:			    	## Check if PROFILE is set, if not set it to "gpu", if set check if it's either "gpu" or "cpu".
	@if [ -z "$(PROFILE)" ]; then echo "PROFILE is not set, launch it with either 'gpu' or 'cpu'"; exit 1; fi
	@if [ "$(PROFILE)" != "gpu" ] && [ "$(PROFILE)" != "cpu" ]; then echo "PROFILE must be either 'gpu' or 'cpu'"; exit 1; fi


.PHONY: config
config: check-target check-profile	## Build the compose project.
	@echo "Printing config for target: $${TARGET} - profile: $(PROFILE)"
	@export PROJECT_VERSION=$$(grep '__version__ =' $$(find ./src -name 'version.py') | cut -d '"' -f 2)
	@$(DOCKER_COMPOSE) config $${ARGS}


.PHONY: build
build: check-target check-profile	## Print the compose configuration.
	@echo "Building images with target: $${TARGET}"
	@export PROJECT_VERSION=$$(grep '__version__ =' $$(find ./src -name 'version.py') | cut -d '"' -f 2)
	@$(DOCKER_COMPOSE) build $${ARGS}


.PHONY: run
run: check-target check-profile		## Start the project.
	@echo "Starting containers with target: $${TARGET}"
	@export PROJECT_VERSION=$$(grep '__version__ =' $$(find ./src -name 'version.py') | cut -d '"' -f 2)
	@$(DOCKER_COMPOSE) up $${ARGS}


.PHONY: stop
stop: check-target check-profile	## Stop the project.
	@echo "Stopping containers with target: $${TARGET}"
	@$(DOCKER_COMPOSE) stop $${ARGS}

.PHONY: stats
stats: check-target check-profile	## Check runtime stats.
	@echo "Checking stats with target: $${TARGET}"
	@$(DOCKER_COMPOSE) stats $${ARGS}


.PHONY: down
down: check-target check-profile	## Kill the project eliminating containers, use ARGS="-v" to remove volumes.
	@echo "Stopping containers with target: $${TARGET}"
	@$(DOCKER_COMPOSE) down $${ARGS}


.PHONY: migrate
migrate: check-venv			## Generate migrations.
	@echo "Symlinking .env file to local.env"
	@if [ ! -f ./envs/local.env ]; then echo "File ./envs/local.env does not exist, please create it"; exit 1; fi
	@ln -sf ./envs/local.env .env
	@docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d $${ARGS} database
	@echo "Waiting for database to start..."
	@echo "Generating migrations..."
	@$(PY_BIN)/alembic revision --autogenerate -m "$${MSG}"


.PHONY: test
test:					## Run tests.
	@echo "Setting up test environment..."
	@ln -sf ./envs/test.env .env
	@echo "Executing containerized tests..."
	@export TARGET=test
	@echo "Running on: $(PROFILE)"
	# if profile is cpu, set the container name to backend-cpu, otherwise backend
	@if [ "$(PROFILE)" = "cpu" ]; then \
	docker compose -p serve-test \
		--profile cpu \
		-f docker-compose.yml \
		-f docker-compose.test.yml up --build \
		--abort-on-container-exit --exit-code-from backend-cpu; \
	else \
		docker compose -p serve-test \
		--profile gpu \
		-f docker-compose.yml \
		-f docker-compose.test.yml up --build \
		--abort-on-container-exit --exit-code-from backend; fi
	@echo "Tearing everything down..."
	@docker compose -p serve-test \
		--profile $(PROFILE) \
        -f docker-compose.yml \
        -f docker-compose.test.yml down -v