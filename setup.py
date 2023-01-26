from setuptools import setup, find_packages

VERSION = '1.4.4.2'
DESCRIPTION = 'scHash package for scRNA-seq data integration'
LONG_DESCRIPTION = 'A package that could integrate atlas-level scRNA-seq datasets with multi-million reference database and annotate tens of thousands of cells within seconds.'

setup(
    name="scHash",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=' ',
    author_email=' ',
    url = 'https://github.com/bowang-lab/scHash',
    keywords = ['RNA-SEQ', 'Atlas', 'INTEGRATION'],
    license='MIT',
    packages=find_packages(),
    install_requires=[
        "torch>=1.0.0",
        "pytorch-lightning>=1.6.5",
        "scipy>=1.8.0",
        "numpy>=1.22.1",
        "scikit-learn>=1.0.2",
        "pandas>=1.1.0",
        "anndata>=0.8.0",
        "scanpy>=1.7",
        "matplotlib>=3.4.3"
    ],
    classifiers= [
        "Development Status :: 3 - Alpha",
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)