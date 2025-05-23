from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="mi_herramienta",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
      package_data={                    
        "mi_herramienta": ["models/*.keras"],
    },
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'katulu=mi_herramienta.cli:main',
        ],
    },
)
