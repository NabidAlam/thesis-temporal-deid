#!/usr/bin/env python3
"""
Setup script for MaskAnyone-Temporal
"""

try:
    from setuptools import setup, find_packages
except ImportError:
    print("setuptools not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "setuptools"])
    from setuptools import setup, find_packages

# Read requirements from requirements.txt
def read_requirements():
    try:
        with open('requirements.txt') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        return []

# Read README for long description
def read_readme():
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "MaskAnyone-Temporal: Temporally-Consistent Privacy Protection for Behavioural-Science Video"

setup(
    name="maskanyone-temporal",
    version="0.1.0",
    author="Md Shahabub Alam",
    author_email="nabid.aust37@gmail.com",
    description="Temporally-Consistent Privacy Protection for Behavioural-Science Video",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/NabidAlam/thesis-temporal-deid",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Video",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
