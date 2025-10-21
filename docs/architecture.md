# file_path: mlx_rl_trainer/docs/architecture.md
# revision_no: 001
# goals_of_writing_code_block: Architecture documentation for the MLX RL Trainer.
# type_of_code_response: add new code
# MLX RL Trainer Architecture

## Overview
The MLX RL Trainer is designed as a modular, production-ready system for reinforcement learning training of language models using Apple's MLX framework.

## Core Principles
- **Plugin Architecture**: Rewards and evaluators register themselves dynamically.
- **Configuration-Driven**: All experiments defined declaratively in YAML files.
- **Algorithm Agnostic**: Core abstractions work with any RL algorithm.
- **Extensible**: New capabilities added without extensive core refactoring.

## Directory Structure
```
mlx_rl_trainer/
├── configs/            # Configuration files (YAML)
├── docs/               # Project documentation
├── scripts/            # Entry-point scripts (train, evaluate, serve, data_preprocessing)
├── src/                # Python source code
│   └── mlx_rl_trainer/
│       ├── core/       # Core infrastructure: trainer, config, checkpointing, managers
│       ├── algorithms/ # RL algorithms (GRPO, PPO)
│       ├── data/       # Data loading and processing
│       ├── evaluation/ # Benchmark evaluators
│       ├── generation/ # Text generation utilities
│       ├── monitoring/ # Logging and metrics
│       ├── rewards/    # Reward function plugins
│       └── utils/      # Common utilities (MLX-specific, text, math, distributed)
└── tests/              # Unit and integration tests
```

## Key Components

### 1. Configuration System (`src/mlx_rl_trainer/core/config.py`)
- **Pydantic-based validation**: Ensures all configuration is type-safe and validated at load time.
- **YAML-driven**: Experiments are defined declaratively in YAML files.
- **Hierarchical**: Supports nested configurations for different aspects (trainer, model, data, rewards, etc.).

### 2. Trainer Base Class (`src/mlx_rl_trainer/core/trainer.py`)
- **Abstract base**: Defines the high-level interface that all training algorithms must implement.
- **Algorithm-agnostic**: Provides common training loop logic, checkpointing, and monitoring integration.
- **Extensible**: Designed to be subclassed by algorithm-specific trainers (e.g., `GRPOTrainer`).

### 3. Reward System (`src/mlx_rl_trainer/rewards/`)
- **Plugin architecture**: Reward functions register themselves using a decorator (`@RewardRegistry.register()`).
- **Composable**: Multiple rewards can be combined with specified weights using `RewardComposer`.
- **Extensible**: New custom reward functions (e.g., for specific task types) are easy to add.

### 4. Evaluation System (`src/mlx_rl_trainer/evaluation/`)
- **Benchmark integration**: Provides built-in support for standard benchmarks (e.g., HumanEval, GSM8K).
- **Customizable**: Allows adding domain-specific evaluators to measure performance against specific criteria.
- **Metrics tracking**: Comprehensive collection and reporting of evaluation metrics.

### 5. Data Pipeline (`src/mlx_rl_trainer/data/`)
- **Flexible loading**: Supports loading data from multiple sources (JSONL, HuggingFace datasets, synthetic).
- **Preprocessing**: Handles text cleaning, tokenization, and formatting.
- **Batching**: Provides efficient data loaders with various sampling strategies.

### 6. Generation Utilities (`src/mlx_rl_trainer/generation/`)
- **Autoregressive generation**: Core logic for generating text sequences from the LLM.
- **Sampling strategies**: Supports various techniques like temperature, top-p, top-k sampling.
- **Caching**: Includes optimized KV cache management (e.g., `PagedKVCache`).

### 7. Monitoring & Checkpointing (`src/mlx_rl_trainer/monitoring/`, `src/mlx_rl_trainer/core/checkpoint_manager.py`)
- **Comprehensive Logging**: Integrates `Rich` for console output, file handlers for full debug trails, and `Weights & Biases` for experiment tracking.
- **Atomic Checkpointing**: Ensures training state is saved reliably and can be resumed safely.
- **Rotation & Best Model**: Automatically manages old checkpoints and tracks the best performing model.

## Extension Points

- **Rewards**: Subclass `BaseReward`
- **Evaluators**: Subclass `BaseEvaluator`
- **Algorithms**: Subclass `BaseTrainer`
- **Data Loaders**: Implement loader interface

