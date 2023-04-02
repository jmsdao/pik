from setuptools import find_packages, setup

setup(
    name="pik",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    entry_points={
        'console_scripts': [
            'gen-answers = cli.generate_answers:main',
        ],
    }
)
