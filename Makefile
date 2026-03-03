# Simple and elegant Makefile derived from the almighty https://github.com/pydantic/pydantic
.DEFAULT_GOAL := help
.DEFAULT_GOAL := help
PROFILE := gpu
DOCKER_COMPOSE := docker compose --profile $(PROFILE) -f docker-compose.yml -f docker-compose.$(TARGET).yml
sources = src tests
.ONESHELL:

.PHONY: .uv  ## Check that uv is installed
.uv:
	@uv -V || echo 'Please install uv: https://docs.astral.sh/uv/getting-started/installation/'

.PHONY: .pre-commit  ## Check that pre-commit is installed, or install it
.pre-commit: .uv
	@uv run pre-commit -V || uv pip install pre-commit

.PHONY: .check-target  # Check if the TARGET variable has been set
.check-target:
	@if [ -z "$(TARGET)" ]; then echo "TARGET is not set, launch the command setting TARGET=dev|prod"; exit 1; fi
	@if [ "$(TARGET)" != "dev" ] && [ "$(TARGET)" != "prod" ]; then echo "TARGET must be either 'dev' or 'prod'"; exit 1; fi
	@echo "Symlinking .env file to $${TARGET}.env"
	@if [ ! -f ./envs/$${TARGET}.env ]; then echo "File ./envs/$${TARGET}.env does not exist, please create it"; exit 1; fi
	@ln -sf ./envs/$${TARGET}.env .env

.PHONY: .check-profile  ## Check if PROFILE is set, if not set it to "gpu", if set check if it's either "gpu" or "cpu".
.check-profile:
	@if [ -z "$(PROFILE)" ]; then echo "PROFILE is not set, launch it with either 'gpu' or 'cpu'"; exit 1; fi
	@if [ "$(PROFILE)" != "gpu" ] && [ "$(PROFILE)" != "cpu" ]; then echo "PROFILE must be either 'gpu' or 'cpu'"; exit 1; fi

.PHONY: install  ## Install the package, dependencies, and pre-commit for local development
install: .uv
	uv sync --frozen --all-extras
	uv run pre-commit install --install-hooks

.PHONY: format  ## Auto-format python source files
format: .uv
	uv run ruff check --fix $(sources)
	uv run ruff format $(sources)

.PHONY: lint  ## Lint python source files
lint: .uv
	uv run ruff check $(sources)
	uv run ruff format --check $(sources)

.PHONY: typecheck  ## Perform type-checking
typecheck: .pre-commit
	uv run pyright src/

.PHONY: config  ## Print the full configuration of the compose project
config: check-target check-profile
	@echo "Printing config for target: $${TARGET} - profile: $(PROFILE)"
	@export PROJECT_VERSION=$$(uv version --short)
	@$(DOCKER_COMPOSE) config $${ARGS}

.PHONY: build  ## Build the compose project
build: check-target check-profile
	@echo "Building images with target: $${TARGET}"
	@export PROJECT_VERSION=$$(uv version --short)
	@$(DOCKER_COMPOSE) build $${ARGS}

.PHONY: run  ## Launch the compose project
run: check-target check-profile
	@echo "Starting containers with target: $${TARGET}"
	@export PROJECT_VERSION=$$(uv version --short)
	@$(DOCKER_COMPOSE) up $${ARGS}

.PHONY: stop  ## Stop the compose project
stop: check-target check-profile
	@echo "Stopping containers with target: $${TARGET}"
	@$(DOCKER_COMPOSE) stop $${ARGS}

.PHONY: stats  ## Check runtime stats of the compose project
stats: check-target check-profile
	@echo "Checking stats with target: $${TARGET}"
	@$(DOCKER_COMPOSE) stats $${ARGS}

.PHONY: down  ## Dismantle containers (and volumes with -v) of the compose project
down: check-target check-profile
	@echo "Stopping containers with target: $${TARGET}"
	@$(DOCKER_COMPOSE) down $${ARGS}

.PHONY: migrate  ## Generate the database migrations.
migrate: check-venv
	@echo "Symlinking .env file to local.env"
	@if [ ! -f ./envs/local.env ]; then echo "File ./envs/local.env does not exist, please create it"; exit 1; fi
	@ln -sf ./envs/local.env .env
	@docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d $${ARGS} database
	@echo "Waiting for database to start..."
	@echo "Generating migrations..."
	@$(PY_BIN)/alembic revision --autogenerate -m "$${MSG}"

.PHONY: test  ## Run unit and integration tests in containers
test:
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
		--abort-on-container-exit --exit-code-from tester; \
	else \
		docker compose -p serve-test \
		--profile gpu \
		-f docker-compose.yml \
		-f docker-compose.test.yml up --build \
		--abort-on-container-exit --exit-code-from tester; fi
	@echo "Tearing everything down..."
	@docker compose -p serve-test \
		--profile $(PROFILE) \
        -f docker-compose.yml \
        -f docker-compose.test.yml down -v

.PHONY: clean ## Clean unused files
clean:
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

.PHONY: release  ## Bump version, create git tag and commit (BUMP=major|minor|patch)
release: .uv
ifndef BUMP
	$(error BUMP is not set. Usage: make release BUMP=major|minor|patch)
endif
	@echo "Current version: $$(uv version)"
	@echo "Bumping $(BUMP) version..."
	@uv version --bump $(BUMP)
	@NEW_VERSION=$$(uv version --short)
	@echo "New version: v$$NEW_VERSION"
	@echo "Creating git commit and tag..."
	@git add .
	@git commit --no-verify -m "Bump version to $$NEW_VERSION"
	@git tag -a "v$$NEW_VERSION" -m "Release v$$NEW_VERSION"
	@git push
	@git push origin tag v$$NEW_VERSION
	@echo ""
	@echo "✓ Version bumped to $$NEW_VERSION"
	@echo "  You can now run 'uv publish' if necessary"

.PHONY: help  ## Display this message
help:
	@grep -E \
		'^.PHONY: .*?## .*$$' $(MAKEFILE_LIST) | \
		grep -v '^.PHONY: \.' | \
		sort | \
		awk 'BEGIN {FS = ".PHONY: |## "}; {printf "\033[36m%-19s\033[0m %s\n", $$2, $$3}'