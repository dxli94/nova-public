# PyFlow

Project by ANU Hybrid System Group.

## Getting Started

First, clone PyFlow-assignment into your workplace by

```
git clone git@gitlab.com:dxli1005/PyFlow-assignment.git
```

### Installing dependencies
#### Option 1 (recommended)
The easiest way to install all the dependencies is by creating a virtual environment using Anaconda. See [Anaconda installation instructions](https://conda.io/docs/user-guide/install/index.html).

After installing Anaconda,
```
cd PyFlow-assignment
conda env create --name PyFlow-assignment --file environment.yml
```
then activate the virtual environment by
```
source activate PyFlow-assignment
```
You will see your prompt is now prefixed with "PyFlow-assignment", which is the virtual environment with dependencies installed.

#### Option 2
In case Anaconda does not work as expected, you need to install the following packages manually.

- [SciPy](https://www.scipy.org/) (1.0.0)
- [NumPy](http://www.numpy.org/) (1.14.2)
- [Cython](http://cython.readthedocs.io/en/latest/src/quickstart/install.html) (0.27.3)
- [matplotlib](https://matplotlib.org/) (2.2.0)
- [CVXOPT](http://cvxopt.org/install/index.html) (1.1.9)
- [pycddlib](http://pycddlib.readthedocs.io/en/latest/) (2.0.0)

### Running

See options:
```
cd dir/src
python main.py --help
```

For example:
```
python main.py --path ../instances/free_ball.txt --dt 0 --horizon 1 --sf 0.1 --opvars 0 1

```

Plot from outfile.out:
```
python Plotter.py --path ../out/outfile.out
```


Tested on Ubuntu 16.04.2 LTS Xenial and macOS High Sierra (10.13.3).
