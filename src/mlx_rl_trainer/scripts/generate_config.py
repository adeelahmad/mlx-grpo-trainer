#!/usr/bin/env python3
"""
Simplified Configuration Generator for MLX RL Trainer

This script provides a streamlined CLI interface for generating optimized default
configurations for the MLX RL Trainer. It simplifies the configuration process by
requiring only the essential parameters (model path and training data path) while
setting all other values to optimized defaults.

Features:
- Minimal required input: only model path and training data path
- Optimized defaults for all other configuration values
- Comprehensive validation and error handling
- Support for different configuration templates (minimal, standard, advanced)
- Automatic directory creation for output paths

Usage:
    python -m mlx_rl_trainer.scripts.generate_config --model-path ./models/my_model --data-path ./data/train.jsonl
    python -m mlx_rl_trainer.scripts.generate_config --model-path ./models/my_model --data-path ./data/train.jsonl --output config.yaml
    python -m mlx_rl_trainer.scripts.generate_config --model-path ./models/my_model --data-path ./data/train.jsonl --template advanced

Author: Roo (Elite AI Programming Assistant)
Version: 1.0.0
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
from rich.console import Console
from rich.panel import Panel

# Initialize rich console for enhanced output
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_minimal_config(model_path: Path, data_path: Path, val_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Create a minimal configuration with only essential parameters.
    
    This configuration includes only the bare minimum settings needed to run
    the trainer, with optimized defaults for everything else.
    """
    config = {
        "trainer": {
            "output_dir": "./outputs",
            "num_training_steps": 10000,
            "learning_rate": 2e-6,
            "ppo_batch_size": 1,
            "num_rollout_samples": 2,
            "grad_accum_steps": 1,
            "use_mixed_precision": True,
            "use_compile": True
        },
        "model": {
            "model_path": str(model_path),
            "use_lora": True,
            "lora_rank": 8,
            "lora_alpha": 16.0
        },
        "data": {
            "train_path": str(data_path),
            "max_prompt_len": 512,
            "max_gen_len": 128
        },
        "rewards": [
            {"name": "semantic_similarity", "weight": 0.7},
            {"name": "tag_structure", "weight": 0.3}
        ],
        "generation": {
            "think_temperature": 0.2,
            "answer_temperature": 0.3,
            "sampling_top_p": 0.6,
            "sampling_top_k": 80
        },
        "monitoring": {
            "use_wandb": False
        },
        "checkpointing": {
            "save_dir": "./checkpoints",
            "save_every": 500,
            "keep_last_n": 2
        },
        "evaluation": []
    }
    
    # Add validation data if provided
    if val_path:
        config["data"]["val_path"] = str(val_path)
    
    return config


def generate_standard_config(model_path: Path, data_path: Path, val_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Create a standard configuration with balanced settings.
    
    This configuration includes a balanced set of features and optimizations
    suitable for most use cases.
    """
    # Start with minimal config
    config = generate_minimal_config(model_path, data_path, val_path)
    
    # Enhance with standard features
    config["trainer"]["num_training_steps"] = 20000
    config["trainer"]["learning_rate"] = 1e-6
    config["trainer"]["grad_accum_steps"] = 4
    
    # Enable W&B for standard config
    config["monitoring"]["use_wandb"] = True
    config["monitoring"]["wandb_project"] = "mlx-rl-trainer"
    
    # Add evaluators
    config["evaluation"] = [
        {"name": "perplexity", "config": {}, "enabled": True},
        {"name": "human_eval", "config": {}, "enabled": True}
    ]
    
    return config


def generate_advanced_config(model_path: Path, data_path: Path, val_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Create an advanced configuration with all features enabled.
    
    This configuration includes all available features and optimizations
    for maximum performance and flexibility.
    """
    # Start with standard config
    config = generate_standard_config(model_path, data_path, val_path)
    
    # Enhance with advanced features
    config["trainer"]["num_training_steps"] = 50000
    config["trainer"]["learning_rate"] = 5e-7
    config["trainer"]["grad_accum_steps"] = 8
    config["trainer"]["use_dual_gradients"] = True
    config["trainer"]["adaptive_gradient_weights"] = True
    
    # Advanced model settings
    config["model"]["lora_rank"] = 16
    config["model"]["lora_alpha"] = 32.0
    
    # Advanced monitoring
    config["monitoring"]["profile_memory"] = True
    config["monitoring"]["profile_compute"] = True
    config["monitoring"]["enable_metrics_server"] = True
    
    # Advanced checkpointing
    config["checkpointing"]["save_every"] = 250
    config["checkpointing"]["keep_last_n"] = 5
    config["checkpointing"]["compression_enabled"] = True
    config["checkpointing"]["async_save"] = True
    
    # Advanced generation settings
    config["generation"]["think_boost_tokens"] = 12
    config["generation"]["think_temperature"] = 0.15
    config["generation"]["answer_temperature"] = 0.25
    config["generation"]["sampling_top_p"] = 0.5
    config["generation"]["sampling_top_k"] = 50
    
    # Enable paged KV cache
    config["use_paged_kv_cache"] = True
    config["kv_cache_block_size"] = 8
    
    # Add more rewards
    config["rewards"] = [
        {"name": "semantic_similarity", "weight": 0.5},
        {"name": "tag_structure", "weight": 0.3},
        {"name": "thinking_quality", "weight": 0.2}
    ]
    
    # Add more evaluators
    config["evaluation"].append({"name": "gsm8k", "config": {}, "enabled": True})
    
    return config


def add_yaml_comments(config_dict: Dict[str, Any], template: str) -> Dict[str, Any]:
    """Add helpful comments to the YAML configuration."""
    commented_config = {
        f"# MLX RL Trainer Configuration ({template} template)": None,
        f"# Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}": None,
        f"# Only model_path and train_path are required; all other values are optimized defaults": None,
        **config_dict
    }
    return {k: v for k, v in commented_config.items() if k.startswith('#') or v is not None}


def generate_optimized_config(
    model_path: Path,
    data_path: Path,
    output_path: Path,
    template: str = "standard",
    val_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Generate an optimized configuration with minimal required input.
    
    Args:
        model_path: Path to the model directory
        data_path: Path to the training data file (JSONL)
        output_path: Path where to save the configuration file
        template: Configuration template to use ('minimal', 'standard', 'advanced')
        val_path: Optional path to validation data file
        
    Returns:
        Generated configuration dictionary
        
    Raises:
        ValueError: If inputs are invalid
        FileNotFoundError: If required paths don't exist
    """
    # Validate inputs
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Training data path does not exist: {data_path}")
    
    if val_path and not val_path.exists():
        raise FileNotFoundError(f"Validation data path does not exist: {val_path}")
    
    # Create configuration based on template
    if template == "minimal":
        config = generate_minimal_config(model_path, data_path, val_path)
    elif template == "advanced":
        config = generate_advanced_config(model_path, data_path, val_path)
    else:  # standard
        config = generate_standard_config(model_path, data_path, val_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata comments
    config_with_comments = add_yaml_comments(config, template)
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(
            config_with_comments,
            f,
            default_flow_style=False,
            sort_keys=False,
            indent=2,
            width=100
        )
    
    logger.info(f"Optimized configuration generated successfully: {output_path}")
    return config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate optimized configuration with minimal required input",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model-path ./models/my_model --data-path ./data/train.jsonl
  %(prog)s --model-path ./models/my_model --data-path ./data/train.jsonl --output config.yaml
  %(prog)s --model-path ./models/my_model --data-path ./data/train.jsonl --template advanced
        """
    )
    
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the model directory (required)"
    )
    
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to the training data file (JSONL) (required)"
    )
    
    parser.add_argument(
        "--val-path",
        type=Path,
        help="Path to the validation data file (JSONL) (optional)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("config.yaml"),
        help="Output path for the configuration file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--template",
        choices=["minimal", "standard", "advanced"],
        default="standard",
        help="Configuration template to use (default: standard)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing configuration file"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point for the script."""
    args = parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if output file exists
    if args.output.exists() and not args.force:
        console.print(
            f"[red]Error: Output file {args.output} already exists. "
            f"Use --force to overwrite.[/red]"
        )
        return 1
    
    try:
        console.print(f"Generating {args.template} configuration...")
        
        # Generate configuration
        generate_optimized_config(
            model_path=args.model_path,
            data_path=args.data_path,
            output_path=args.output,
            template=args.template,
            val_path=args.val_path
        )
        
        # Display success message
        console.print(
            Panel(
                f"[green]✓[/green] Optimized configuration generated successfully!\n\n"
                f"[bold]Output:[/bold] [cyan]{args.output}[/cyan]\n"
                f"[bold]Template:[/bold] {args.template}\n"
                f"[bold]Model Path:[/bold] {args.model_path}\n"
                f"[bold]Training Data:[/bold] {args.data_path}",
                title="Configuration Generated",
                border_style="green"
            )
        )
        
        return 0
        
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if args.verbose:
            import traceback
            console.print_exception()
        return 1


if __name__ == "__main__":
    sys.exit(main())