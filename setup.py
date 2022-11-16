"""Setups diffusion for text."""

import setuptools


def setup_package():
    setuptools.setup(
        name="sdlm",
        version="0.0.1",
        description="Simplex diffusion for language models.",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        classifiers=[
            "Intended Audience :: Science/Research",
            "Development Status :: 3 - Alpha",
            "License :: OSI Approved :: Apache Software License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        keywords="",
        url="https://github.com/allenai/diffusion",
        author="Allen Institute for Artificial Intelligence",
        author_email="rabeehk@allenai.org",
        license="Apache",
        packages=setuptools.find_packages(
            exclude=["tests", "docs", "scripts", "examples"],
        ),
        dependency_links=[
            "https://download.pytorch.org/whl/torch_stable.html",
        ],
        python_requires=">=3.7",
    )


if __name__ == "__main__":
    setup_package()
