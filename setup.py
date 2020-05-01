#!/usr/bin/env python

import setuptools
from distutils.core import setup

setup(name='npeuclid',
      py_modules=['npeuclid'],
      version='0.1',
      license='lgpl-2.1',
      description='Fast 2D and 3D vector geometry module',
      author='Dov Grobgeld',
      author_email='dov.grobgeld@gmail.com',
      url='http://github.com/dov/npeuclid',
      download_url = 'https://github.com/dov/npeuclid/archive/v0.1.tar.gz',
      keywords = ['math','geometry'],
      install_requires=[            
          'numpy',
      ],
      setup_requires=['wheel'],
      classifiers = [
          "Programming Language :: Python",
          "Programming Language :: Python :: 3",
          "Development Status :: 3 - Alpha",
          "Environment :: Other Environment",
          "Intended Audience :: Developers",
          "License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)",
          "Operating System :: OS Independent",
          "Topic :: Software Development :: Libraries :: Python Modules",
          "Topic :: Scientific/Engineering :: Mathematics",
          ],
      )

