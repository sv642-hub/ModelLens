from setuptools import setup, find_packages

setup(
    name="modellens",
    version="0.1.0",
    description="An open-source interpretability toolkit for PyTorch neural networks",
    author="6600 Neural Networks - Sebastian/Vinny/Fareeza/Sharanya/Jeff",
    packages=find_packages(exclude=("tests", "tests.*", "examples", "examples.*")),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
    ],
    entry_points={
        "console_scripts": [
            "modellens-gradio=app.main:main",
        ],
    },
    extras_require={
        "viz": [
            "matplotlib>=3.7.0",
            "plotly>=5.15.0",
            "pandas>=2.0.0",
            "nbformat>=4.2.0",
        ],
        "app": [
            "gradio>=4.0.0",
            "plotly>=5.15.0",
            "pandas>=2.0.0",
            "nbformat>=4.2.0",
        ],
        "dev": [],
    },
)
