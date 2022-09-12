from setuptools import setup
from adfp import __version__

setup(name='adfp',
      version=__version__,
      license='MIT License',
      install_requires=['numpy'],
      description='auto differential framework',
      author='Takumi Enomoto',
      author_email='eno.sleepy.zzz.zz.z@gmail.com',
      url='',
      packages=['adfp'],
      )