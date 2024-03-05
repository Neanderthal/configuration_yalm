from setuptools import setup, find_packages

setup(
    name="configuration_yalm",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "sentencepiece",
        "transformers",
        "torch",
        "six",
        # Add any dependencies required by your package
    ],
)
