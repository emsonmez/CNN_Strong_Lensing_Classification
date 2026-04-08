from setuptools import find_packages, setup

setup(
    name="CNN_Strong_Lensing_Classification",
    version="0.1.0",
    description="",
    author="Emrecan Michael Sonmez",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "jupyter",
        "pytest",
        "pytest-cov",
        "pytest-mock",
        "pyswip",
        "scikit-learn",
        "scikit-optimize",
    ],
    extras_require={
        "dev": [
            "black",
            "ruff",
            "mypy",
        ]
    },
    include_package_data=True,
)
