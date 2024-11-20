from setuptools import setup, find_namespace_packages

setup(
    name="financial-crisis-detection",
    version="0.1.0",
    packages=find_namespace_packages(include=["src*"]),
    package_dir={"": "."},
    install_requires=[
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "transformers>=4.30.0",
        "torch-geometric>=2.3.0",
        "wandb>=0.15.0",
        "hydra-core>=1.3.0",
        "einops>=0.6.0",
        "torchtyping>=0.1.4",
        "pytorch-metric-learning>=2.1.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
    ],
    python_requires=">=3.8",
)
