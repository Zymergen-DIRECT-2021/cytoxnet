import setuptools
from os import path
import cytoxnet

here = path.abspath(path.dirname(__file__))
AUTHORS = """
Nida Janulaitis
Rory McGuire
Evan Komp
"""

# Get the long description from the README file
with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

if __name__ == "__main__":
    setuptools.setup(
        name='cytoxnet',
        version=cytoxnet.__version__,
        author=AUTHORS,
        project_urls={
            'Source': 'https://github.com/Zymergen-DIRECT-2021/cytoxnet',
        },
        description=
        'Tools for predicting cytotoxicity if compounds to microbes.',
        long_description=long_description,
        include_package_data=False, #no data yet, True if we want to include data
        keywords=[
            'Machine Learning', 'Synthetic Biology', 'Cytotocicity'
        ],
        license='MIT',
        packages=setuptools.find_packages(exclude="tests"),
        scripts = [], #if we want to include shell scripts we make in the install
        classifiers=[
            'Development Status :: 1 - Planning',
            'Environment :: Console',
            'Operating System :: OS Independant',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
        ],
        zip_safe=False,
    )
