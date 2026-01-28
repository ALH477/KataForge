# Dojo Manager Makefile
# Copyright © 2026 DeMoD LLC. All rights reserved.

.PHONY: help setup test lint format clean docker deploy

# Variables
PYTHON := python3
PYTEST := pytest
BLACK := black
RUFF := ruff
PROJECT_NAME := dojo-manager
VERSION := 0.1.0

# Help
help:
	@echo "Dojo Manager - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Initial setup (Nix + dependencies)"
	@echo "  make setup-rocm     - Setup with ROCm/AMD GPU support"
	@echo "  make setup-cuda     - Setup with CUDA/NVIDIA GPU support"
	@echo ""
	@echo "Development:"
	@echo "  make test           - Run all tests"
	@echo "  make test-unit      - Run unit tests only"
	@echo "  make test-e2e       - Run end-to-end tests"
	@echo "  make lint           - Run linters"
	@echo "  make format         - Format code"
	@echo "  make validate       - Validate system"
	@echo ""
	@echo "Training:"
	@echo "  make preprocess     - Preprocess videos"
	@echo "  make extract-poses  - Extract poses from videos"
	@echo "  make train          - Train all models"
	@echo ""
	@echo "Deployment:"
	@echo "  make docker         - Build Docker images"
	@echo "  make deploy-local   - Deploy locally"
	@echo "  make deploy-prod    - Deploy to production"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean          - Clean generated files"
	@echo "  make docs           - Generate documentation"

# Setup
setup:
	@echo "Setting up Dojo Manager..."
	nix develop --command bash -c "poetry install"
	@echo "✓ Setup complete"

setup-rocm:
	@echo "Setting up with ROCm support..."
	cp flake.nix flake-rocm.nix.bak || true
	@echo "✓ ROCm setup complete"

setup-cuda:
	@echo "Setting up with CUDA support..."
	@echo "✓ CUDA setup complete"

# Testing
test:
	$(PYTEST) tests/ -v --cov=dojo_manager --cov-report=term-missing

test-unit:
	$(PYTEST) tests/unit/ -v

test-integration:
	$(PYTEST) tests/integration/ -v

test-e2e:
	$(PYTEST) tests/e2e/ -v

# Code Quality
lint:
	$(RUFF) check dojo_manager/
	$(PYTHON) -m mypy dojo_manager/

format:
	$(BLACK) dojo_manager/ tests/
	$(RUFF) check --fix dojo_manager/

validate:
	$(PYTHON) scripts/system_validator.py
	$(PYTHON) scripts/config_validator.py config/framework16.yaml

# Training Pipeline
preprocess:
	dojo-manager video batch-preprocess data/raw/ data/processed/ --workers 8

extract-poses:
	dojo-manager pose batch-extract data/processed/ data/poses/ --workers 4

calculate-biomechanics:
	dojo-manager biomechanics batch-calculate data/poses/ data/metrics/ --workers 8

split-data:
	dojo-manager data split data/poses/ --train-ratio 0.7 --output-dir data/splits/

train:
	dojo-manager train all-models \
		--config config/framework16.yaml \
		--data-dir data/splits/ \
		--output-dir models/$(VERSION)

train-graphsage:
	dojo-manager train graphsage --config config/framework16.yaml

train-form-assessor:
	dojo-manager train form-assessor --config config/framework16.yaml

train-style-encoder:
	dojo-manager train style-encoder --config config/framework16.yaml

# Docker
docker:
	nix build .#docker
	docker load < result

docker-rocm:
	nix build .#docker --arg rocmSupport true
	docker load < result

# Deployment
deploy-local:
	dojo-manager server start --host localhost --port 8000

deploy-staging:
	kubectl apply -f k8s/staging/
	kubectl rollout status deployment/dojo-manager -n staging

deploy-prod:
	kubectl apply -f k8s/production/
	kubectl rollout status deployment/dojo-manager -n production

# Documentation
docs:
	@echo "Generating documentation..."
	@echo "See docs/ directory for comprehensive guides"

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage .mypy_cache
	rm -rf build/ dist/
	@echo "✓ Cleaned"

clean-data:
	@echo "⚠️  This will delete all processed data!"
	@read -p "Continue? (y/N): " confirm && [ "$$confirm" = "y" ]
	rm -rf data/processed/* data/poses/* data/metrics/*
	@echo "✓ Data cleaned"

# GPU Testing
test-gpu:
	$(PYTHON) -c "from dojo_manager.core.gpu_utils import test_gpu_operations; test_gpu_operations()"

test-gpu-rocm:
	$(PYTHON) scripts/test_gpu.py

# Monitoring
monitor-gpu:
	watch -n 1 rocm-smi

monitor-training:
	tail -f logs/training.log

# Quick Start
quickstart:
	./scripts/framework16-quickstart.sh

# Version
version:
	@echo "Dojo Manager v$(VERSION)"
	@echo "ROCm support: Yes (175W optimized)"
	@echo "Hardware target: Framework 16 (AMD RX 7700S)"
