# PyFlow

Project by ANU Hybrid System Group.

## Getting Started

First, clone PyFlow-assignment into your workplace by

```
git clone git@gitlab.com:dxli-private/HyPyFlow.git
```

### Installing dependencies
#### Option 1 (recommended)
The easiest way to install all the dependencies is by creating a virtual environment using Anaconda. See [Anaconda installation instructions](https://conda.io/docs/user-guide/install/index.html).

After installing Anaconda,
```
cd HyPyFlow/
conda env create --name HyPy --file environment.yml
```
then activate the virtual environment by
```
source activate HyPy
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

Try affine analyzer, see options
```
cd HyPyFlow/src
python affine_analyzer.py --help
```

For example:
```
python affine_analyzer.py --path ../instances/single_mode_affine_instances/free_ball.txt --dt 0 --horizon 1 --sampling_time 0.1 --opvars 0 1

```

Plot from outfile.out:
```
python Plotter.py --path ../out/outfile.out
```

Try nonlinear analyzer (dev.),
 
1. choose the path to instance in main() of non_linear_analyzer.py; e.g. 
```
 ../instances/non_linear_instances/vanderpol.json
```
2. change time horizon, sampling time, direction type, initial states, etc. in the corresponding json file.
3. try
```
python non_linear_analyzer.py
```
4. Plot from outfile.out:
```
python Plotter.py --path ../out/outfile.out
```
 

Tested on Ubuntu 16.04.2 LTS Xenial and macOS High Sierra (10.13.3).
