This program allows the determination of the time step at which a particle crosses a given plane and/or desorbs 
from a liquid. The latter is based on the loss of all coordinating solvent molecules. This can be useful in
conjuction with my other project, https://github.com/thekannman/z_energy_trace, that uses the output from this
program to average energy and force traces relative to the time steps from this or similar programs.

THIS PROGRAM HAS NOT BEEN THOROUGHLY TESTED!!!

Current limitations include:
* Only trajectories from the GROMACS simulation package can be used as input.
* The program currently only allows determination of desorption as defined above, not of crossing a plane.

The following libraries are required:
* GROMACS XTC library for reading position/velocity files. 
  http://www.gromacs.org/Developer_Zone/Programming_Guide/XTC_Library
* The Armadillo library for matrix calculations. http://arma.sourceforge.net/
* The Boost libraries for input option parsing, 4D tensor calculations, lexical casting, and output formatting. 
  http://www.boost.org/
