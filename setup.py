from setuptools import setup, find_packages

setup(
    name='triforce',
    version='0.1.0',
    packages=['triforce_lib'] + find_packages(),
    entry_points={
        'console_scripts': [
            'triforce=triforce:main',
        ],
    },
)
