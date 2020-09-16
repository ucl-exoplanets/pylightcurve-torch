import setuptools

with open('README.md', 'r') as dh:
    long_description = dh.read()

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
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    setup_requires=['setuptools-git-versioning'],
    python_requires='>=3.7',
    install_requires=['torch']
)
