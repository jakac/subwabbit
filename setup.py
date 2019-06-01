import runpy
import sys

from setuptools import find_packages, Command, setup
from setuptools.command.test import test as TestCommand


__version__ = runpy.run_path("subwabbit/version.py")["__version__"]


class PyLint(Command):

    description = 'Runs code quality check.'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        self.distribution.fetch_build_eggs([
            'astroid==1.5.3',
            'pylint==1.7.4',
        ])
        from pylint.lint import Run
        Run([
            '--rcfile', 'pylintrc',
            './subwabbit'
            ])

class PyMyPy(Command):

    description = 'Runs mypy check.'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        self.distribution.fetch_build_eggs([
            'mypy==0.641',
        ])
        from mypy.main import main
        main(
            script_path='',
            args=[
                '--ignore-missing-imports',
                '.',
            ]
        )

class PyTest(TestCommand):

    user_options = [
        ('pytest-args=', 'a', "Arguments to pass to py.test"),
    ]

    def run(self):
        TestCommand.run(self)

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


with open('README.rst') as f:
    long_description = f.read()


setup(
    name="subwabbit",
    version=__version__,
    author='Matej Jakimov',
    author_email="matej.jakimov@gmail.com",
    description=("Fast Python Vowpal Wabbit wrapper"),
    long_description=long_description,
    long_description_content_type='text/x-rst',
    license="BSD",
    url='https://github.com/jakac/subwabbit',
    classifiers=[
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: BSD License',
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
    ],
    package_data={
        '': ['CHANGELOG.md']
    },
    packages=['subwabbit'],
    install_requires=[
        'mypy-lang',
        'typing',
    ],
    tests_require=[
        'mypy==0.521',
        'pylint',
        'pytest',
        'pytest-benchmark',
    ],
    test_suite='tests',
    cmdclass={
        'lint': PyLint,
        'mypy': PyMyPy,
        'test': PyTest,
    },
)
