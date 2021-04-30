import setuptools

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setuptools.setup(
    name='rvfn',
    version='1.0.0',
    packages=setuptools.find_packages(),
    install_requires=requirements
)
