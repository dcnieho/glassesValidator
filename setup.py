import setuptools
import runpy

info = runpy.run_path("src/glassesValidator/version.py")

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

required = []

# get required packages
for line in requirements:
    required.append(line)

# generate reference board image
#validationSetup = utils.getValidationSetup(configDir)
#reference = utils.Reference(configDir, validationSetup)

setuptools.setup(
    name=info['__title__'],
    version=info['__version__'],
    author=info['__author__'],
    author_email=info['__email__'],
    description=info['__description__'],
    long_description=readme,
    long_description_content_type="text/markdown",
    url=info['__url__'],
    project_urls={
        "Source Code": info['__url__'],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license=license,
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=required,
    entry_points={
        "gui_scripts": [
            "glassesValidator = glassesValidator.GUI:run",
        ],
    },
)
