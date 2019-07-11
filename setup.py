from setuptools import setup

setup(name='O2py',
      version='0.1',
      description='An interactive visualization of the 2d O(2) model with matplotlib',
      author='Manes Hornung',
      author_email='hornung@itp.unibe.ch',
      #license='MIT',
      packages=['O2py'],
      install_requires=['numpy', 'scipy', 'matplotlib', 'numba'],
      extras_require={'lic':['licpy', 'tensorflow']},
      zip_safe=False)
