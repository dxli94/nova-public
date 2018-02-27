# PyHyBase

Project by ANU Hybrid System Group.

## Getting Started

### Prerequisites

- [SciPy](https://www.scipy.org/)
- [NumPy](http://www.numpy.org/)
- [Matplotlib](https://matplotlib.org/) and/or [plotutils](https://www.gnu.org/software/plotutils/)
- [pycddlib](http://pycddlib.readthedocs.io/en/latest/)
- [Cython](http://cython.readthedocs.io/en/latest/src/quickstart/install.html)


Tested on Ubuntu 16.04.2 LTS xenial.

### Running

See options:
```
cd dir/src
python3 main.py --help
```

For example:
```
python3 main.py --path ../instances/instance_3.txt --dt 0 --horizon 1 --sf 0.1 --output 1 --opvars 0 1

```

Plot from outfile.out:
```
./plot.sh
```
