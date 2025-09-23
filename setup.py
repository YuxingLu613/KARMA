"""Setup script for KARMA package."""

from setuptools import setup, find_packages
import pathlib

# Get the long description from the README file
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

# Read version from the package
def get_version():
    import sys
    sys.path.insert(0, str(here / "karma"))
    try:
        from karma import __version__
        return __version__
    except ImportError:
        return "1.0.0"

setup(
    name="karma-nlp",
    version=get_version(),
    description="Multi-Agent LLMs for Automated Knowledge Graph Enrichment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YuxingLu613/KARMA",
    author="Yuxing Lu",
    author_email="yxlu0613@gmail.com",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="nlp, knowledge-graph, biomedical, multi-agent, llm, named-entity-recognition, relationship-extraction",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
        "PyPDF2>=3.0.0",
        "typing-extensions>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "all": [
            "spacy>=3.4.0",
            "networkx>=2.8.0",
            "matplotlib>=3.5.0",
            "pandas>=1.4.0",
            "jupyter>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "karma=karma.cli:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/YuxingLu613/KARMA/issues",
        "Source": "https://github.com/YuxingLu613/KARMA",
        "Documentation": "https://karma-nlp.readthedocs.io/",
    },
    include_package_data=True,
    zip_safe=False,
)