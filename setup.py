# file_path: mlx_rl_trainer/setup.py
# revision_no: 001
# goals_of_writing_code_block: Define the package structure and dependencies for the MLX RL Trainer.
# type_of_code_response: add new code
"""Setup configuration for the MLX RL Trainer package."""
from setuptools import setup, find_packages

setup(
    name="mlx_rl_trainer",
    version="0.6.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "mlx-grpo-trainer=mlx_rl_trainer.scripts.train:main",
            "mlx-grpo-evaluate=mlx_rl_trainer.scripts.evaluate:main",
            "mlx-grpo-preprocess=mlx_rl_trainer.scripts.data_preprocessing:main",
            "mlx-grpo-generate-config=mlx_rl_trainer.scripts.generate_config:main",
        ],
    },
    install_requires=[
        "mlx>=0.5.0",
        "mlx-lm>=0.10.0",  # Required for mlx_lm.utils.load, TokenizerWrapper, etc.
        "pydantic>=2.0",
        "numpy>=1.24.0",
        "datasets>=2.14.0",  # HuggingFace datasets library
        "pyyaml>=6.0",  # For config files
        "rich>=13.0.0",  # For enhanced logging and progress bars
        "tqdm>=4.60.0",  # For console progress bars
        "aiofiles>=22.0.0",  # For asynchronous file I/O
        "scikit-learn>=1.3.0",  # For TF-IDF in reward functions
        "wandb>=0.15.0",  # For experiment tracking
        "psutil>=5.9.0",  # For system monitoring
        "seaborn>=0.12.0",  # For enhanced visualizations
    ],
    extras_require={
        "dev": [
            "pandas>=2.0.0",  # For metrics plotting
            "matplotlib>=3.7.0",  # For metrics plotting
            "scipy>=1.10.0",  # For advanced statistics
            "pillow>=9.5.0",  # For image processing
            "pytest",  # For unit and integration testing
            "pytest-asyncio",  # For testing async code
        ]
    },
    python_requires=">=3.9",
)
