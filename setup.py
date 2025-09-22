#!/usr/bin/env python3
from setuptools import setup, find_packages
from pathlib import Path
from typing import List

def parse_requirements(filename: str = "clarifi3d-requirements.txt") -> List[str]:
    """
    Load package dependencies from a pip requirements file.
    """
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Missing requirements file: {filename}")
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

def load_readme(path: str = "README.md") -> str:
    readme = Path(path)
    return readme.read_text(encoding="utf-8") if readme.exists() else ""

setup(
    name="clarifi3d",
    version="0.1.0",
    description="Fast, GPU-accelerated 3D segmentation pipeline for peroxisomes, nuclei, and cell boundaries",
    long_description=load_readme(),
    long_description_content_type="text/markdown",
    author="Fred Mast",
    author_email="Fred.Mast@SeattleChildrens.org",
    url="https://github.com/seattlechildrens/clarifi3d",
    python_requires=">=3.9,<4",
    packages=find_packages(exclude=["tests*", "docs*", "notebooks*", "scripts*"]),
    install_requires=parse_requirements(),
    entry_points={
        "console_scripts": [
            "clarifi3d=clarifi3d.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Unix",
        "License :: OSI Approved :: MIT License",
    ],
    keywords=[
        "3D segmentation", "peroxisomes", "deep learning", "microscopy", "cell biology", "HPC", "GPU", "CuPy", "PyTorch"
    ],
)