from setuptools import setup, find_packages

setup(
    name="prompt-metrics",
    version="0.1.0",
    description="Scripts and experiments for scoring and testing LLM prompt responses",
    author="Olusegun Akinbo",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "openai>=1.0.0",
        "anthropic>=0.20.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0"],
    },
)
