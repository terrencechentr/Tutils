from pathlib import Path

from setuptools import find_packages, setup

BASE_DIR = Path(__file__).parent
README_FILE = BASE_DIR / "README.md"
README = README_FILE.read_text(encoding="utf-8") if README_FILE.exists() else ""

setup(
    name="tutils",
    version="0.1.0",
    description="Lightweight utilities for ML and data-science workflows",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your@email.com",
    url="https://github.com/yourname/tutils",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "torch",
        "colorlog",
        "tqdm",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)