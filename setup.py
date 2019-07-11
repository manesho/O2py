from setuptools import setup

setup(name='O2py',
      version='0.1',
      description='An interactive visualization of the 2d O(2) model with matplotlib',
      author='Manes Hornung',
      author_email='hornung@itp.unibe.ch',
      #license='MIT',
      packages=['O2py'],
<<<<<<< HEAD
      install_requires=['numpy', 'scipy', 'matplotlib', 'numba'],
      extras_require={'lic':['licpy', 'tensorflow']},
=======
      entry_points="""
          [console_scripts]
          o2py=O2py.cli:cli
      """,
      install_requires=['numpy', 'scipy', 'matplotlib', 'numba', 'click'],
>>>>>>> f82753d109ec4d5216ad446e4c74aecbded25bde
      zip_safe=False)
