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
