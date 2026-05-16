from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='movie-recommendation-system',
    version='1.0.0',
    author='shreyptl1157',
    author_email='your-email@example.com',
    description='A comprehensive movie recommendation system with multiple algorithms',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/shreyptl1157/movie-recommendation-system',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.9',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'movie-rec=src.cli:main',
        ],
    },
)
