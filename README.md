# Direct stellarator coil design using global optimization: application to a comprehensive exploration of quasi-axisymmetric devices
The examples directory contain scripts for executing phase II and III of the workflow in

*Direct stellarator coil design using global optimization: application to a comprehensive exploration of quasi-axisymmetric devices*, A. Giuliani, Arxiv

## Background
The algorithm during phase II described in

*Direct computation of magnetic surfaces in Boozer coordinates and coil optimization for quasisymmetry*, A Giuliani, F Wechsung, A Cerfon, G Stadler, M Landreman, Journal of Plasma Physics 88 (4), 905880401

takes coil sets and optimizes them for nested flux surfaces and quasisymmetry on a toroidal volume.  The algorithm in phase III described in

*Direct stellarator coil optimization for nested magnetic surfaces with precise quasi-symmetry*, A Giuliani, F Wechsung, A Cerfon, G Stadler, M Landreman, Physics of Plasmas 30 (4)

polishes these configurations for an accurate approximation of quasisymmetry.  

## Installation
To use this code, first clone the repository including all its submodules, via

    git clone --recursive

Next, best practice is to generate a virtual environment and install PyPlasmaOpt there:

    cd simsopt
    python -m venv venv
    source venv/bin/activate
    cd thirdparty/LinkingNumber/; mkdir build; cd build; cmake ..; make; cd ../../
    pip install -e .

## Running the scripts
Optimizing a coil set obtained from a near-axis optimization in phase II, then III is done by calling

    ./ls1.py 
    ./ex.py 1

The above scripts optimize on a surface of aspect ratio 20.  Then, surfaces on aspect ratio 10 can be targeted:

    ./ls.py 2
    ./ex.py 2

and so on..

    ./ls.py 3
    ./ex.py 3
