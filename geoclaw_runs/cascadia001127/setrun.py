""" 
Module to set up run time parameters for Clawpack -- AMRClaw code.

The values set in the function setrun are then written out to data files
that will be read in by the Fortran code.

""" 

import os
import numpy as np

topodir = os.path.abspath('../topo')

dtopodir = '../make_dtopo'
dtopo_name = 'cascadia001127.dtt3'   # desired earthquake deformation

#------------------------------
def setrun(claw_pkg='geoclaw'):
#------------------------------
    
    """ 
    Define the parameters used for running Clawpack.

    INPUT:
        claw_pkg expected to be "geoclaw" for this setrun.

    OUTPUT:
        rundata - object of class ClawRunData 
    
    """ 
    
    from clawpack.clawutil import data 
    
    
    assert claw_pkg.lower() == 'geoclaw',  "Expected claw_pkg = 'geoclaw'"

    num_dim = 2
    rundata = data.ClawRunData(claw_pkg, num_dim)


    #------------------------------------------------------------------
    # Problem-specific parameters to be written to setprob.data:
    #------------------------------------------------------------------


    #------------------------------------------------------------------
    # GeoClaw specific parameters:
    #------------------------------------------------------------------
    rundata = setgeo(rundata)
    
    #------------------------------------------------------------------
    # Standard Clawpack parameters to be written to claw.data:
    #------------------------------------------------------------------

    clawdata = rundata.clawdata  # initialized when rundata instantiated


    # Set single grid parameters first.
    # See below for AMR parameters.


    # ---------------
    # Spatial domain:
    # ---------------

    # Number of space dimensions:
    clawdata.num_dim = num_dim
    
    # Lower and upper edge of computational domain:
    clawdata.lower[0] = -132           # xlower
    clawdata.upper[0] = -122.          # xupper
    clawdata.lower[1] = 38.           # ylower
    clawdata.upper[1] = 51.           # yupper
    
    # Number of grid cells:
    clawdata.num_cells[0] = 40      # mx
    clawdata.num_cells[1] = 52     # my
    

    # ---------------
    # Size of system:
    # ---------------

    # Number of equations in the system:
    clawdata.num_eqn = 3

    # Number of auxiliary variables in the aux array (initialized in setaux)
    clawdata.num_aux = 3
    
    # Index of aux array corresponding to capacity function, if there is one:
    clawdata.capa_index = 2
    
    
    # -------------
    # Initial time:
    # -------------

    clawdata.t0 = 0.0
    

    # Restart from checkpoint file of a previous run?
    # Note: If restarting, you must also change the Makefile to set:
    #    RESTART = True
    # If restarting, t0 above should be from original run, and the
    # restart_file 'fort.chkNNNNN' specified below should be in 
    # the OUTDIR indicated in Makefile.

    clawdata.restart = False               # True to restart from prior results
    clawdata.restart_file = 'fort.chk00060'  # File to use for restart data
    
    
    # -------------
    # Output times:
    #--------------

    # Specify at what times the results should be written to fort.q files.
    # Note that the time integration stops after the final output time.
 
    clawdata.output_style = 2
 
    if clawdata.output_style==1:
        # Output ntimes frames at equally spaced times up to tfinal:
        # Can specify num_output_times = 0 for no output
        clawdata.num_output_times = 24
        clawdata.tfinal = 6*3600.
        clawdata.output_t0 = True  # output at initial (or restart) time?
        
    elif clawdata.output_style == 2:
        # Specify a list or numpy array of output times:
        # Include t0 if you want output at the initial time.
        clawdata.output_times =  list(np.linspace(0,5*60,11)) + \
                                 list(range(15*60,60*60,15*60)) + \
                                 list(range(60*60,3*3600+1,30*60))
 
    elif clawdata.output_style == 3:
        # Output every step_interval timesteps over total_steps timesteps:
        clawdata.output_step_interval = 1
        clawdata.total_steps = 1
        clawdata.output_t0 = True  # output at initial (or restart) time?
        

    clawdata.output_format = 'binary'      # 'ascii', 'binary', 'netcdf'

    clawdata.output_q_components = 'all'   # could be list such as [True,True]
    clawdata.output_aux_components = 'none'  # could be list
    clawdata.output_aux_onlyonce = True    # output aux arrays only at t0
    

    # ---------------------------------------------------
    # Verbosity of messages to screen during integration:  
    # ---------------------------------------------------

    # The current t, dt, and cfl will be printed every time step
    # at AMR levels <= verbosity.  Set verbosity = 0 for no printing.
    #   (E.g. verbosity == 2 means print only on levels 1 and 2.)
    clawdata.verbosity = 1
    
    

    # --------------
    # Time stepping:
    # --------------

    # if dt_variable==True:  variable time steps used based on cfl_desired,
    # if dt_variable==Falseixed time steps dt = dt_initial always used.
    clawdata.dt_variable = True
    
    # Initial time step for variable dt.  
    # (If dt_variable==0 then dt=dt_initial for all steps)
    clawdata.dt_initial = 0.1
    
    # Max time step to be allowed if variable dt used:
    clawdata.dt_max = 1e+99
    
    # Desired Courant number if variable dt used 
    clawdata.cfl_desired = 0.9
    # max Courant number to allow without retaking step with a smaller dt:
    clawdata.cfl_max = 1.0
    
    # Maximum number of time steps to allow between output times:
    clawdata.steps_max = 5000


    # ------------------
    # Method to be used:
    # ------------------

    # Order of accuracy:  1 => Godunov,  2 => Lax-Wendroff plus limiters
    clawdata.order = 2
    
    # Use dimensional splitting? (not yet available for AMR)
    clawdata.dimensional_split = 'unsplit'
    
    # For unsplit method, transverse_waves can be 
    #  0 or 'none'      ==> donor cell (only normal solver used)
    #  1 or 'increment' ==> corner transport of waves
    #  2 or 'all'       ==> corner transport of 2nd order corrections too
    clawdata.transverse_waves = 2
    
    
    # Number of waves in the Riemann solution:
    clawdata.num_waves = 3
    
    # List of limiters to use for each wave family:  
    # Required:  len(limiter) == num_waves
    # Some options:
    #   0 or 'none'     ==> no limiter (Lax-Wendroff)
    #   1 or 'minmod'   ==> minmod
    #   2 or 'superbee' ==> superbee
    #   3 or 'vanleer'  ==> van Leer
    #   4 or 'mc'       ==> MC limiter
    clawdata.limiter = ['vanleer', 'vanleer', 'vanleer']
    
    clawdata.use_fwaves = True    # True ==> use f-wave version of algorithms
    
    # Source terms splitting:
    #   src_split == 0 or 'none'    ==> no source term (src routine never called)
    #   src_split == 1 or 'godunov' ==> Godunov (1st order) splitting used, 
    #   src_split == 2 or 'strang'  ==> Strang (2nd order) splitting used,  not recommended.
    clawdata.source_split = 1
    
    
    # --------------------
    # Boundary conditions:
    # --------------------

    # Number of ghost cells (usually 2)
    clawdata.num_ghost = 2
    
    # Choice of BCs at xlower and xupper:
    #   0 or 'user'     => user specified (must modify bcNamr.f to use this option)
    #   1 or 'extrap'   => extrapolation (non-reflecting outflow)
    #   2 or 'periodic' => periodic (must specify this at both boundaries)
    #   3 or 'wall'     => solid wall for systems where q(2) is normal velocity
    
    clawdata.bc_lower[0] = 'extrap'   # at xlower
    clawdata.bc_upper[0] = 'extrap'   # at xupper

    clawdata.bc_lower[1] = 'extrap'   # at ylower
    clawdata.bc_upper[1] = 'extrap'   # at yupper
                  
       
    # ---------------
    # Gauges:
    # ---------------

    gauges = rundata.gaugedata.gauges
    # for gauges append lines of the form  [gaugeno, x, y, t1, t2]

    rundata.gaugedata.min_time_increment = 5.
    #rundata.gaugedata.q_out_fields = [0,3]
    
    # DART buoys:
    gauges.append([46404, -128.736, 45.857, 0., 1.e10])
    gauges.append([46407, -128.832, 42.682, 0., 1.e10])
    gauges.append([46419, -129.619, 48.796, 0., 1.e10])
    

    # gauges across SJdF:
    gauges.append([700, -124.5945,48.4038, 0., 1.e10])
    gauges.append([701, -124.5745,48.4320, 0., 1.e10])
    gauges.append([702, -124.5516,48.4648, 0., 1.e10])
    gauges.append([703, -124.5268,48.4988, 0., 1.e10])
    gauges.append([704, -124.5027,48.5331, 0., 1.e10])
    gauges.append([710, -124.0957,48.2336, 0., 1.e10])
    gauges.append([711, -124.0735,48.2738, 0., 1.e10])
    gauges.append([712, -124.0543,48.3081, 0., 1.e10])
    gauges.append([713, -124.0251,48.3586, 0., 1.e10])
    gauges.append([714, -124.0045,48.3974, 0., 1.e10])


    t0_gauge = 0.0;
    # gauges near Discovery Bay:
    gauges.append([900, -122.91  ,48.11  , t0_gauge, 1.e10])
    gauges.append([901, -122.86  ,48.0   , t0_gauge, 1.e10])
    gauges.append([902, -122.8467,48.0135, t0_gauge, 1.e10])

    
    # gauges near Whidbey Island, San Juan Island and Victoria:
    gauges.append([910, -122.775 ,48.23  , t0_gauge, 1.e10])
    gauges.append([911, -122.7457,48.1669, t0_gauge, 1.e10])
    gauges.append([912, -122.7331,48.1062, t0_gauge, 1.e10])
    gauges.append([913, -122.6401,48.0932, t0_gauge, 1.e10])

    gauges.append([920, -122.7362,48.4027, t0_gauge, 1.e10])
    gauges.append([921, -122.9962,48.4449, t0_gauge, 1.e10])
    gauges.append([922, -123.1761,48.4645, t0_gauge, 1.e10])
    gauges.append([923, -123.3912,48.3998, t0_gauge, 1.e10])

    # gauges from Carrie
    gauges.append([1000, -124.6406,48.3519, \
            t0_gauge, 1.e10])
    gauges.append([1001, -123.7058,48.1584, \
            t0_gauge, 1.e10])
    gauges.append([1002, -122.8892,47.9916, \
            t0_gauge, 1.e10])
    gauges.append([1003, -122.8894,47.9905, \
            t0_gauge, 1.e10])
    gauges.append([1004, -122.8901,47.9888, \
            t0_gauge, 1.e10])
    gauges.append([1005, -122.7209,48.3043, \
            t0_gauge, 1.e10])
    gauges.append([1006, -122.1817,48.0368, \
            t0_gauge, 1.e10])

    gauges.append([1010, -124.1078,48.2097, \
            t0_gauge, 1.e10])
    gauges.append([1011, -122.7471,48.0283, \
            t0_gauge, 1.e10])
    gauges.append([1012, -122.5278,47.9111, \
            t0_gauge, 1.e10])
    gauges.append([1013, -122.7055,48.3158, \
            t0_gauge, 1.e10])
    gauges.append([1014, -122.6781,48.5029, \
            t0_gauge, 1.e10])
    gauges.append([1015, -122.3432,47.5583, \
            t0_gauge, 1.e10])
    gauges.append([1016, -122.3551,47.2583, \
            t0_gauge, 1.e10])


    gauges.append([1020, -122.8419,47.436, \
            t0_gauge, 1.e10])
    gauges.append([1021, -122.8440,47.4349, \
            t0_gauge, 1.e10])
    gauges.append([1022, -122.8416,47.4475, \
            t0_gauge, 1.e10])

    # gauges across SJdF for BC:
    x1,y1 = -124.4, 48.3
    x2,y2 = -124.4, 48.51
    yy = np.linspace(y1,y2,17)
    xx = np.linspace(x1,x2,17)
    for j in range(len(yy)):
        gaugeno = 500+j
        gauges.append([gaugeno, xx[j], yy[j], 0., 1.e10])

    # gauges across Admiralty Inlet
    x1,y1 = -122.664, 48.03
    x2,y2 = -122.61, 48.03
    yy = np.linspace(y1,y2,9)
    xx = np.linspace(x1,x2,9)
    for j in range(len(yy)):
        gaugeno = 600+j
        gauges.append([gaugeno, xx[j], yy[j], 0., 1.e10])

                  
    # --------------
    # Checkpointing:
    # --------------

    # Specify when checkpoint files should be created that can be
    # used to restart a computation.

    clawdata.checkpt_style = 2

    if clawdata.checkpt_style == 0:
      # Do not checkpoint at all
      pass

    elif clawdata.checkpt_style == 1:
      # Checkpoint only at tfinal.
      pass

    elif clawdata.checkpt_style == 2:
      # Specify a list of checkpoint times.  
      clawdata.checkpt_times = [3600, 7200]

    elif clawdata.checkpt_style == 3:
      # Checkpoint every checkpt_interval timesteps (on Level 1)
      # and at the final time.
      clawdata.checkpt_interval = 5

    

    # ---------------
    # AMR parameters:   (written to amr.data)
    # ---------------
    amrdata = rundata.amrdata

    # max number of refinement levels:
    amrdata.amr_levels_max = 5

    # List of refinement ratios at each level (length at least amr_level_max-1)
    #   15', 3', 30", 6", 2", 2/3" [5, 6, 5, 3, 3]
    amrdata.refinement_ratios_x = [5, 6, 5, 3, 3]
    amrdata.refinement_ratios_y = [5, 6, 5, 3, 3]
    amrdata.refinement_ratios_t = [5, 6, 5, 3, 3]


    # Specify type of each aux variable in amrdata.auxtype.
    # This must be a list of length num_aux, each element of which is one of:
    #   'center',  'capacity', 'xleft', or 'yleft'  (see documentation).
    amrdata.aux_type = ['center', 'capacity', 'yleft']


    # Flag for refinement based on Richardson error estimater:
    amrdata.flag_richardson = False    # use Richardson?
    amrdata.flag_richardson_tol = 1.0  # Richardson tolerance
    
    # Flag for refinement using routine flag2refine:
    amrdata.flag2refine = True      # use this?
    amrdata.flag2refine_tol = 0.5  # tolerance used in this routine
    # Note: in geoclaw the refinement tolerance is set as wave_tolerance below 
    # and flag2refine_tol is unused!

    # steps to take on each level L between regriddings of level L+1:
    amrdata.regrid_interval = 4

    # width of buffer zone around flagged points:
    # (typically the same as regrid_interval so waves don't escape):
    amrdata.regrid_buffer_width  = 4

    # clustering alg. cutoff for (# flagged pts) / (total # of cells refined)
    # (closer to 1.0 => more small grids may be needed to cover flagged cells)
    amrdata.clustering_cutoff = 0.7

    # print info about each regridding up to this level:
    amrdata.verbosity_regrid = 0      


    # ---------------
    # Regions:
    # ---------------
    regions = rundata.regiondata.regions 
    # to specify regions of refinement append lines of the form
    #  [minlevel,maxlevel,t1,t2,x1,x2,y1,y2]

    # refinement to level 2 allowed over entire domain:
    regions.append([1, 2, 0., 1e9, clawdata.lower[0]-0.1, \
        clawdata.upper[0]+0.1,clawdata.lower[1]-0.1,clawdata.upper[1]+0.1])

    # over short time around earthquake source:
    regions.append([3, 3, 0., 100, -126, -123.5, 39, 50.5]) #earthquake source 
    #regions.append([4, 4, 0., 100, -126, -123.5, 47.5, 48.5]) #earthquake source 

    # along coast:
    regions.append([1, 3, 0., 1e9, -130, -123.8, 40, 50.])

    # in Strait:
    rundata.regiondata.regions.append([3, 4, 0., 1e9, -125.12,-122.2,47.9,48.79])
        
    # around Discovery Bay:
    turnon_DB=45.0*60.      
    rundata.regiondata.regions.append([5, 5,turnon_DB,1e9,\
        -122.96,-122.81,47.98,48.14])

    # It turns out these three regions does not improve the results
    # at gauges inside them
    # # near Fidalgo Island, Lopez Island and Port Angeles
    # turnon_FI=45.0*60.      
    # rundata.regiondata.regions.append([5, 5,turnon_FI,1e9,\
    #     -122.8125,-122.6073,48.0116,48.4705])
    # turnon_LI=35.0*60.      
    # rundata.regiondata.regions.append([5, 5,turnon_LI,1e9,\
    #     -123.4599,-122.6300,48.3717,48.5418])
    # turnon_PA=30.0*60.      
    # rundata.regiondata.regions.append([5, 5,turnon_PA,1e9,\
    #         -123.7363,-123.6691,48.1546,48.1791])

    # fgmax grid region:
    # rundata.regiondata.regions.append([6, 6, 1.5*3600., 1e9,\
    #     -122.9, -122.83, 47.98, 48.02])




    #  ----- For developers ----- 
    # Toggle debugging print statements:
    amrdata.dprint = False      # print domain flags
    amrdata.eprint = False      # print err est flags
    amrdata.edebug = False      # even more err est flags
    amrdata.gprint = False      # grid bisection/clustering
    amrdata.nprint = False      # proper nesting output
    amrdata.pprint = False      # proj. of tagged points
    amrdata.rprint = False      # print regridding summary
    amrdata.sprint = False      # space/memory output
    amrdata.tprint = False      # time step reporting each level
    amrdata.uprint = False      # update/upbnd reporting
    
    return rundata

    # end of function setrun
    # ----------------------


#-------------------
def setgeo(rundata):
#-------------------
    """
    Set GeoClaw specific runtime parameters.
    """

    try:
        geo_data = rundata.geo_data
    except:
        print("*** Error, this rundata has no geo_data attribute")
        raise AttributeError("Missing geo_data attribute")

    # == Physics ==
    geo_data.gravity = 9.81
    geo_data.coordinate_system =  2
    geo_data.earth_radius = 6367500.0

    # == Forcing Options
    geo_data.coriolis_forcing = False

    # == Algorithm and Initial Conditions ==
    tide_stage = 77.
    geo_data.sea_level = (tide_stage - 77.)/100.    #  m relative to MHW
    # ******Set in run_tests.py ******

    geo_data.dry_tolerance = 0.001
    geo_data.friction_forcing = True
    geo_data.manning_coefficient = 0.025
    geo_data.friction_depth = 100.0

    # Refinement settings
    refinement_data = rundata.refinement_data
    refinement_data.variable_dt_refinement_ratios = True
    refinement_data.wave_tolerance = 0.02
    refinement_data.deep_depth = 100.0
    refinement_data.max_level_deep = 4

    # == settopo.data values ==


    topofiles = rundata.topo_data.topofiles
    # for topography, append lines of the form
    #    [topotype, minlevel, maxlevel, t1, t2, fname]

    topofiles.append([3, 1, 1, 0., 1.e10, \
            os.path.join(topodir, 'etopo1_-163_-122_38_63.asc')])
    topofiles.append([3, 1, 1, 0., 1.e10,\
            os.path.join(topodir, 'SJdF_2sec_center.asc')])
    topofiles.append([3, 1, 1, 0., 1.e10,\
            os.path.join(topodir, 'PT_2sec_center.asc')])
    topofiles.append([3, 1, 1, 0., 1.e10,\
            os.path.join(topodir, 'PS_2sec_center.asc')])
    topofiles.append([3, 1, 1, 0., 1.e10,\
           os.path.join(topodir, 'DiscoveryBay_1_3s_center.asc')])


    # == setdtopo.data values ==
    rundata.dtopo_data.dtopofiles = []
    dtopofiles = rundata.dtopo_data.dtopofiles
    # for moving topography, append lines of the form :  
    #   [topotype, minlevel,maxlevel,fname]

    dtopotype  = 3
    dtopofiles.append([dtopotype, 3, 3, os.path.join(dtopodir,dtopo_name)])
    rundata.dtopo_data.dt_max_dtopo = 20. # deformation over 240 sec


    # == setqinit.data values ==
    rundata.qinit_data.qinit_type =  0
    rundata.qinit_data.qinitfiles = []
    qinitfiles = rundata.qinit_data.qinitfiles 
    # for qinit perturbations, append lines of the form: (<= 1 allowed for now!)
    #   [minlev, maxlev, fname]

    # == fixedgrids.data values ==
    rundata.fixed_grid_data.fixedgrids = []
    fixedgrids = rundata.fixed_grid_data.fixedgrids
    # for fixed grids append lines of the form
    # [t1,t2,noutput,x1,x2,y1,y2,xpoints,ypoints,\
    #  ioutarrivaltimes,ioutsurfacemax]

    # == fgmax.data values ==
    fgmax_files = rundata.fgmax_data.fgmax_files
    # for fixed grids append to this list names of any fgmax input files
    #fgmax_files.append('fgmax_grid_1.txt')  
    #rundata.fgmax_data.num_fgmax_val = 2  # record depth and velocities



    return rundata
    # end of function setgeo
    # ----------------------

if __name__ == '__main__':
    # Set up run-time parameters and write all data files.
    import sys
    rundata = setrun(*sys.argv[1:])
    rundata.write()
    
    from clawpack.geoclaw import kmltools
    kmltools.make_input_data_kmls(rundata)

