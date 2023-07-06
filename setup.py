"""Install setup."""
import setuptools


setup_requirements = ['pytest-runner']
test_requirements = ['pytest']


setuptools.setup(
    name="bain_tnc",
    version="0.0.1",

    author="Samuel Edet",
    description="Bain and TNC hackathon",

    packages=setuptools.find_packages(include=['src']),
    install_requires=[],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    tests_require=test_requirements
)




