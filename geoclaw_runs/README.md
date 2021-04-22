# geoclaw_runs

Sample code to use GeoClaw for creating seafloor deformations and
running tsunami simulations to capture the gauge data used in training
and testing the ML algorithms.

The GeoClaw code itself must be obtained and installed as described at
http://www.clawpack.org/installing.html.   

The results used in the paper were obtained using Clawpack version 5.7.0. 
Later versions should work too.  See http://www.clawpack.org/releases.html

## Organization:

### Subdirectory make_topo 

Contains a Jupyter notebook illustrating how to make a dtopo (seafloor
deformation) file from one of the fakequakes.  Data is included for
one fakequake realization; others can be downloaded from
http://doi.org/10.5281/zenodo.59943

 - CSZ_fault_geometry.ipynb	 Jupyter notebook
 - CSZ_fault_geometry.html	 Rendered version of notebook output and plots	
 - cascadia30.mshout         Triangulation of CSZ fault geometry
 - _cascadia.001127.rupt     Rupture file for fakequake realization #1127


### Subdirectory cascadia001127  

Contains the GeoClaw code required to run one tsunami simulation
and collect time series at the gauges used in the paper (and many
others).  This code can be easily modified to run any other event
by simply changing the dtopofile specification in setrun.py.

The files included are the basic ones needed to specify the run parameters and 
the desired plots:

 - Makefile
 - setrun.py
 - setplot.py
 

### Subdirectory topo

Running the GeoClaw code requires topography DEM files that should be
placed in this directory (if elsewhere, change the path in setrun.py).
The topography files needed can be downloaded from the repository
http://depts.washington.edu/ptha/topo/

 - wget_topo.sh   bash script to download topography files
 
