import setuptools


def parse_long_description(filename):
    """ read long description file """
    with open(filename, 'r') as f:
        return f.read()


def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


long_description = parse_long_description('README.md')

install_reqs = parse_requirements("requirements.txt")

setuptools.setup(
    name="structvib",
    version="0.0.0",
    author="Emmanuel CIEREN",
    author_email="ecieren@eurobios.com",
    url="",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=install_reqs,
    packages=setuptools.find_packages(),
    include_package_data=True,
    setup_requires=['wheel'],
)
