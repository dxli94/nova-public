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
cd NOVA
conda env create --name nova --file environment.yml
```
then activate the virtual environment by
```
source activate nova
```
You will see your prompt is now prefixed with "PyFlow-assignment", which is the virtual environment with dependencies installed.

#### Option 2
In case Anaconda is not available or does not work as expected, you could install the following packages manually.

- [SciPy](https://www.scipy.org/) (1.0.0)
- [NumPy](http://www.numpy.org/) (1.14.2)
- [Cython](http://cython.readthedocs.io/en/latest/src/quickstart/install.html) (0.27.3)
- [matplotlib](https://matplotlib.org/) (2.2.0)
- [CVXOPT](http://cvxopt.org/install/index.html) (1.1.9)
- [pycddlib](http://pycddlib.readthedocs.io/en/latest/) (2.0.0)
- [sympy](http://docs.sympy.org/latest/install.html) (1.1.1)

They should all be available if you are using pip.

### Running

Try nonlinear analyzer (dev.),
 
Choose the path to instance in main() of non_linear_analyzer.py; e.g. 
```
 ../instances/non_linear_instances/vanderpol.json
```
change time horizon, sampling time, direction type, initial states, etc. in the corresponding json file.
Try
```
python non_linear_analyzer.py
```
Plot from outfile.out:
```
python Plotter.py --path ../out/outfile.out
```
 

Tested on Ubuntu 16.04.2 LTS Xenial and macOS High Sierra (10.13.3).
