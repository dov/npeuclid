#!/usr/bin/env python

'''
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

from distutils.core import setup

setup(name='npeuclid',
      version='0.01',
      description='Fast 2D and 3D vector geometry module',
      author='Dov Grobgeld',
      author_email='dov.grobgeld@gmail.com',
      url='http://github.com/dov/npeuclid',
      py_modules=['npeuclid'],
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

