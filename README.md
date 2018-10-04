# NOVA

Project by ANU Hybrid System Group.

## Getting Started

First, clone NOVA into your workplace by

```
git clone git@gitlab.com:dxli-private/nova.git
```

### Installing dependencies
#### Option 1 (recommended)
The easiest way to install all the dependencies is by creating a virtual environment using Anaconda. See [Anaconda installation instructions](https://conda.io/docs/user-guide/install/index.html).

After installing Anaconda,
```
cd nova
conda env create --name nova --file environment.yml
```

Install gmpy2 seperatly,
```
pip install gmpy2==2.1.0a1 --ignore-installed --no-binary ":all:"
```
then activate the virtual environment by
```
source activate nova
```
You will see your prompt is now prefixed with "nova", which is the virtual environment with dependencies installed.

#### Option 2
In case Anaconda is not available or does not work as expected, you could install the following packages manually.

- [SciPy](https://www.scipy.org/) (1.0.0)
- [NumPy](http://www.numpy.org/) (1.14.2)
- [Cython](http://cython.readthedocs.io/en/latest/src/quickstart/install.html) (0.27.3)
- [matplotlib](https://matplotlib.org/) (2.2.0)
- [CVXOPT](http://cvxopt.org/install/index.html) (1.1.9)
- [sympy](http://docs.sympy.org/latest/install.html) (1.1.1)
- [pplpy](https://gitlab.com/videlec/pplpy) (0.7)

They should all be available if you are using pip.

### Running

Try affine analyzer,
 
Choose the path to instance in nova/instance/single_mode_affine_instances; e.g. 
```
 ../instances/single_mode_affine_instances/free_ball.json
```
change time horizon, sampling time, direction type, initial states, etc. in the corresponding json file.
Try
```
python affine_linear_analyzer.py
```
Plot from outfile.out on dimension 0 and 1 (indexed from 0):
```
./plot_polygon.sh ../out/outfile.out 0 1  
```
 

Tested on Ubuntu 16.04.2 LTS Xenial and macOS High Sierra (10.13.3).
