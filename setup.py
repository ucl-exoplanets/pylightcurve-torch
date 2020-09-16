import subprocess
import warnings

import setuptools

with open('README.md', 'r') as dh:
    long_description = dh.read()

classifiers = [
    "License :: OSI Approved :: GNU General Public License (GPL)",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Operating System :: OS Independent",
]

setup_requires = ['setuptools-git-versioning'],
try:
    subprocess.call(['git', '--version'])
except:
    warnings.warn("git not available on platform, setuptools-versioning can't be used")
    setup_requires = []

setuptools.setup(
    name="pylightcurve-torch",
    version_config=True,
    author="Mario Morvan",
    author_email="mario.morvan.18@ucl.ac.uk",
    url="https://github.com/ucl-exoplanets/pylightcurve-torch.git",
    description="Transit modelling in Pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=classifiers,
    setup_requires=setup_requires,
    python_requires='>=3.6',
    install_requires=['torch']
)
