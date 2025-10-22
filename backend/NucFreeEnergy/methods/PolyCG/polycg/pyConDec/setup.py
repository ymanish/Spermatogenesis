from setuptools import setup

setup(name='pyConDec',
      version='0.0.1',
      description='Module for conditional decorators',
      url='https://github.com/eskoruppa/pyConDec',
      author='Enrico Skoruppa',
      author_email='enrico dot skoruppa at gmail dot com',
      license='GNU2',
      packages=['pycondec'],
      package_dir={
            'pycondec': 'pycondec',
      },
      zip_safe=False) 
