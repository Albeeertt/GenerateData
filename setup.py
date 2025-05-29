from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="generate_data",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
      package_data={                    
        "generate_data": ["models/*.keras", "models/*.pt"],
    },
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'KMAsgec=generate_data.cli:main',
        ],
    },
)
