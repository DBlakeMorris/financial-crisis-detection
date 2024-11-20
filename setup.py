from setuptools import setup, find_packages

setup(
    name="financial-crisis-detection",
    version="0.1.0",
    packages=find_packages(),
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
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced financial crisis detection using multi-modal deep learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
)
