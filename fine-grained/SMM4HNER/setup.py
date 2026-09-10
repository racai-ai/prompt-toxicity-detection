from setuptools import setup, find_packages

setup(
    name="smm4hner",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "hner=cli:main",
        ],
    },
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "adapters>=0.2.0",
        "gliner>=0.1.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "seqeval>=1.2.2",
        "dspy",
        "gepa",
        "datasets",
    ],
    python_requires=">=3.10",
)
