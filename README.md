# NOVA
<img src="/docs/logo_text_medium.png"  width="500" height="144">


A tool for **NO**nlinear System **V**erification and **A**nalysis, developed by:
- [Dongxu Li](https://dxli94.github.io/) - dongxuli1005 at gmail dot com
- [Stanley Bak](http://stanleybak.com/) - stanleybak at gmail dot com
- [Sergiy Bogomolov](https://www.sergiybogomolov.com/) - bogom dot s at gmail dot com

## What is NOVA?
NOVA is a verification tool for autonomous continuous system models with nonlinear and linear ODEs. The tool was mostly written by [Dongxu Li](https://dxli94.github.io/) with input from [Stanley Bak](http://stanleybak.com/).

NOVA achieves verification by performing reachability analysis. That is, computing an abstraction of reachable states, called *flowpipe*, given a system model and an initial set. NOVA adopts novel techniques such as *dynamics scaling* to maintain the reachability analysis precision. Examples of the flowpipes constructed for Van der Pol, Brusselator and Buckling Column systems are shown below.

Van der Pol| Brusselator             |  Buckling Column
:-------------------------:|:-------------------------:|:-------------------------:
<img src="/docs/vanderpol.png"  width="560" height="200">| <img src="/docs/bruss.png"  width="560" height="200">  | <img src="/docs/buckle.png"  width="660" height="200"> |

## Setup
First, clone NOVA into your workplace by

```
git clone git@gitlab.com:dxli1005/nova-public.git
```

You can setup NOVA with a few steps. These instructions are for Ubuntu Linux, and may (not) work on other systems.

1. There is a custom C++ interface to Kodiak for use in NOVA that you need to compile. See `nova-public/src/utils/pykodiak/README.md` for details.
Essentially, you need to get a static library of Kodiak (v-2.0) and then run `make` (the Makefile is in the same folder as `README.md`).
This will produce `pykodiak.so`.
2. Install Python packages:

    i. Virtual Environment via Anaconda (Recommended)

    The easiest way to install the packages is by creating a virtual environment using Anaconda. See [Anaconda installation instructions](https://conda.io/docs/user-guide/install/index.html).

    After installing Anaconda, change `conda_path` in `install_nova.sh` to the Anaconda root
    e.g. `home/user/anaconda3`;
    then create a new environment with the required packages installed using the following commands:
    ```
    cd nova-public
    sudo chmod +x ./install_nova.sh
    ./install_nova.sh
    ```

    then activate the virtual environment by
    ```
    source activate nova
    ```
    You will see your prompt is now prefixed with "nova", which is the virtual environment with packages installed.
    (This can be automated by installing [autoenv](https://github.com/zpm-zsh/autoenv) plugin into your terminal and providing a `.env` file
    in the root directory).

    You can deactivate the current virtual environment by
    ```angular2html
    source deactivate
    ```

    ii. Manual installation:
    In case Anaconda is not available or does not work as expected, you could install the following packages manually.

    - [SciPy](https://www.scipy.org/) (1.0.0)
    - [NumPy](http://www.numpy.org/) (1.14.2)
    - [Cython](http://cython.readthedocs.io/en/latest/src/quickstart/install.html) (0.27.3)
    - [matplotlib](https://matplotlib.org/) (2.2.0)
    - [CVXOPT](http://cvxopt.org/install/index.html) (1.1.9)
    - [sympy](http://docs.sympy.org/latest/install.html) (1.1.1)
    - [gmpy2](https://gmpy2.readthedocs.io/en/latest/) (2.1.0a1) by `pip install gmpy2==2.1.0a1 --ignore-installed --no-binary ":all:"`
    - [pplpy](https://gitlab.com/videlec/pplpy) (0.7)

    They should all be available if you are using pip.

## Getting Started
Models in NOVA are defined in Python code.
To try some examples, first go to `nova-public/src/nova_runner`, select and import the model in `src/examples/`. Then run
```angular2html
cd nova-public/src
python nova_runner.py
```
After the computation finishes, the projection of the flowpipe can be found in the corresponding directory under
`out/imgs`.
