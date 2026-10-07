IMAGE ?= panoramic-features
DATA_DIR ?= $(CURDIR)/data
OUTPUT_DIR ?= $(CURDIR)/output

.PHONY: install test lint format docker-build docker-run

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check . && ruff format --check .

format:
	ruff format . && ruff check --fix .

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	mkdir -p $(OUTPUT_DIR)
	docker run --rm --user $$(id -u):$$(id -g) \
		-v $(DATA_DIR):/data:ro -v $(OUTPUT_DIR):/output $(IMAGE)
