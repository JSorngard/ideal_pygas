#!/bin/bash
python -m numpy.f2py -c src/gaslib.f90 -m gaslib --f90flags="-fopenmp" -lgomp
