from setuptools import setup
from pyautodiff import __version__

setup(name='pyautodiff',
      version=__version__,
      license='MIT License',
      install_requires=['numpy'],
      description='auto differential framework',
      author='Takumi Enomoto',
      author_email='eno.sleepy.zzz.zz.z@gmail.com',
      url='',
      packages=['pyautodiff'],
      )