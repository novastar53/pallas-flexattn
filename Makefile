.PHONY: install clean build dev lint format test benchmark help

# Default Python version
PYTHON_VERSION ?= 3.12

# Detect platform for JAX extras
UNAME_M := $(shell uname -m)
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    JAX_PLATFORM = cpu
else ifeq ($(UNAME_S),Linux)
    ifeq ($(shell command -v nvidia-smi > /dev/null 2>&1 && echo yes),yes)
        JAX_PLATFORM = gpu
    else
        JAX_PLATFORM = cpu
    endif
else
    JAX_PLATFORM = cpu
endif

# Install dependencies
install:
	@command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh)
	uv sync --extra $(JAX_PLATFORM)

# Install development dependencies
dev: install
	uv sync --extra dev --extra $(JAX_PLATFORM)

# Clean build artifacts
clean:
	rm -rf uv.lock
	rm -rf build/
	rm -rf dist/
	rm -rf src/pallas_flexattn.egg-info/
	rm -rf .pytest_cache/
	rm -rf src/pallas_flexattn/__pycache__/
	rm -rf tests/__pycache__/
	rm -rf .coverage
	rm -rf .mypy_cache/
	rm -rf .venv/

# Run tests
test:
	uv run pytest tests/ -v

# Run linting
lint:
	uv run ruff check src/pallas_flexattn/ tests/

# Format code
format:
	uv run ruff format src/pallas_flexattn/ tests/

# Build package
build:
	uv build

# Run benchmark
benchmark:
	uv run python -m pallas_flexattn.benchmark --seq-len 2048

# Run kernel tuner
tuner:
	uv run python -m pallas_flexattn.kernel_tuner --seq-len 2048

# Show help
help:
	@echo "Available commands:"
	@echo "  make install   - Install dependencies"
	@echo "  make dev       - Install dev dependencies"
	@echo "  make test      - Run tests"
	@echo "  make lint      - Run linting"
	@echo "  make format    - Format code"
	@echo "  make clean     - Clean build artifacts"
	@echo "  make build     - Build package"
	@echo "  make benchmark - Run performance benchmark"
	@echo "  make tuner     - Show optimal kernel parameters"
	@echo "  make help      - Show this help"
