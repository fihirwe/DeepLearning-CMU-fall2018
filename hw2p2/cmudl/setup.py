from setuptools import setup

setup(name='cmudl',
      version='0.1',
      description='Utility for CMU Deep Learning (11-785)',
      author='Ryan Brigden',
      author_email='rbrigden@cmu.edu',
      license='MIT',
      packages=['cmudl'],
      zip_safe=False,
      scripts=['bin/cmudl'],
      install_requires=[
            'boto3',
            'numpy',
            'requests'
      ]
      )