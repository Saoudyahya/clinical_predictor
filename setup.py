from setuptools import setup, find_packages
import os

# Read README if it exists
long_description = "A clinical prediction system using machine learning"
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

setup(
    name="clinical-predictor",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A clinical prediction system using machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/clinical-predictor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "coverage",
            "black",
            "isort",
            "pylint",
            "bandit",
            "flake8",
        ],
        "api": [
            "flask",
        ],
        "ui": [
            "streamlit",
        ],
    },
)