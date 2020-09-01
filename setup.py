import setuptools

with open('README.md', 'r') as dh:
    long_description = dh.read()

setuptools.setup(
    name="pylightcurve-torch",
    version_config=True,
    setup_requires=['setuptools-git-versioning'],
    author="Mario Morvan",
    author_email="mario.morvan.18@ucl.ac.uk",
    description="Transit modelling in Pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mariomorvan/pylightcurve-torch.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
