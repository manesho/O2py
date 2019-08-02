# O2py
An interactive visualization of the 2d O(2) model.

![Alt text](ex.png?raw=true "Example")

## Setup
Directly via pip:
```console
$ pip install git+https://github.com/manesho/O2py.git 
```
If you want to enable the line integral convolution visualization:
```console
$ pip install git+https://github.com/manesho/O2py.git#egg=O2py[lic]
```

The half vortex graph visualization depends on [graph-tool](https://graph-tool.skewed.de/)
which is based on the [Boost Graph Library](https://www.boost.org/doc/libs/1_70_0/libs/graph/doc/index.html) and therefore cannot be installed via pip. To enable the half vortex graph visualization, you have to [install graph-tool](https://git.skewed.de/count0/graph-tool/wikis/installation-instructions) seperately.

## Usage 
For a simple visualization, run the command line script:

```bash
o2py interact -l 100 --beta 1.1199
```

or within an interactive Python session:
```python
import O2py
O2py.interactiveo2plot(l=100, beta = 1.1199)
```
pay attention to the key bindings printed out to play around.
