"""
Setup script for UPIR package.

This is a clean room implementation based on the TD Commons disclosure:
https://www.tdcommons.org/dpubs_series/8852/

Author: Subhadip Mitra
License: Apache 2.0
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="upir",
    version="0.1.0",
    description="Universal Plan Intermediate Representation - Formal verification, synthesis, and optimization for distributed systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Subhadip Mitra",
    author_email="contact@subhadipmitra.com",
    url="https://github.com/bassrehab/upir",
    license="Apache-2.0",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    python_requires=">=3.9",
    install_requires=[
        "z3-solver>=4.12.2",
        "numpy>=1.24.3,<2.0.0",
        "scikit-learn>=1.3.0,<2.0.0",
    ],
    extras_require={
        "gcp": [
            "google-cloud-bigquery>=3.11.0",
            "google-cloud-pubsub>=2.18.0",
            "google-cloud-storage>=2.10.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "mypy>=1.5.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "ruff>=0.0.285",
            "types-setuptools",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.1.0",
            "mkdocstrings[python]>=0.22.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Code Generators",
        "Topic :: System :: Distributed Computing",
    ],
    keywords="formal-verification program-synthesis distributed-systems smt cegis reinforcement-learning",
    project_urls={
        "Documentation": "https://upir.subhadipmitra.com",
        "Source": "https://github.com/bassrehab/upir",
        "Issues": "https://github.com/bassrehab/upir/issues",
    },
    package_data={
        "upir": ["py.typed"],
    },
    include_package_data=True,
    zip_safe=False,
)
