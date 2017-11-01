from setuptools import setup

setup(
    name='qsim',
    version='0.1',

    install_requires=['numpy>=1.12.1',
                      'matplotlib>=2.0.1'],

    author='Natalie Pearson',
    author_email='npearson@phys.ethz.ch',

    description=("Package for doing quantum simulations using "
                 "various backends."),

    license='MIT',

    packages=['qsim'],

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Licence :: MIT Licence',
        'Programming Language :: Python :: 3.5'
    ],

    keywords='quantum simulation runga kutta mps exact diagonalisation',

    url='https://github.com/nataliejpg/Qsim',

    python_requires='>=3',

)
