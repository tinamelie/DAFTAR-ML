from setuptools import setup, find_packages
from pathlib import Path
from typing import List, Dict


def read_requirements() -> List[str]:
    """Return a list of requirements from requirements.txt (if present)."""
    req_path = Path(__file__).with_name("requirements.txt")
    if req_path.exists():
        with req_path.open() as f:
            return [line.strip() for line in f.read().splitlines() if line.strip() and not line.startswith('#')]
    return []


# Extract version without importing the package (to avoid dependency issues)
about: Dict[str, str] = {}
init_path = Path(__file__).parent / "daftar" / "__init__.py"
if init_path.exists():
    with init_path.open() as f:
        for line in f:
            if line.startswith("__version__"):
                exec(line, about)
                break

description = (
    "Data-Agnostic Feature-Target Analysis & Ranking Machine Learning Pipeline "
    "(DAFTAR-ML)."
)

setup(
    name="daftar-ml",
    version=about.get("__version__", "0.1.0"),
    author="Melie",
    author_email="drtinamelie@gmail.com",
    url="https://github.com/tmelie/DAFTAR-ML",
    description=description,
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="See LICENSE file",
    python_requires=">=3.8",
    packages=find_packages(exclude=("tests", "results", "examples")),
    py_modules=["preprocess", "cv_calculator", "run_daftar"],
    include_package_data=True,
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "daftar=daftar.cli:main",
            "run-daftar=daftar.cli:main",
            "daftar-preprocess=preprocess:main",
            "daftar-cv=cv_calculator:main",
        ]
    },
    # Will add appropriate classifiers later
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
