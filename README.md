# ideal_pygas
Simulates an ideal gas in two dimensions with the initial state given by the Maxwell-Boltzmann distribution.
Can output the result either in a window or as an mp4 file (using ffmpeg).

A fortran library is available to speed up some computation, but is not necessary.
To compile it you must have a compiler capable of compiling fortran, numpy, as well as the openmp libraries on you computer. If you do you only have to run  
. compile.bat  
and then add the --fortran flag when calling the program.
