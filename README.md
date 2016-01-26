# TetFEM3D.jl #

A (spectral) finite element package to simulate the wave equation. Contains

* 1D wave propagation and parameter estimation, including hessian-vector computations.
* 2D wave propagation using Triangles using a modified quadrature rule to maintain convergence.
* 3D wave propagation using Tetrahedra using a modified quadrature rule to maintain convergence.

## Installation

`TetFEM3D.jl` should work with Julia 0.4.x, along with the following Julia packages:

    MAT
    Winston
    Match
    PyPlot
    NPZ
    HDF5
    JLD
    ProfileView
    Devectorize
    ProgressMeter
    SymPy
    Interpolations
    Grid
    Optim
    NLopt
    Ipopt
    MathProgBase
    JuMP

Place this list into your `~/.julia/v0.4/REQUIRE` and call `Pkg.update()` within the Julia repl to install them all. `PyCall` is easiest with the anaconda python distribution and the path can be set with

    ENV["PYTHON"] = "... path of the python program you want ..."
    Pkg.build("PyCall")

The code is structured as a number of modules, such as `Mesh`, `FEM`, or `TimeStepping`, each of which covers its specific area and is loaded using either the `using` `import` commands. Unfortunately, Julia doesn't load modules in the current directory unless you add the following to your `$HOME/.juliarc.jl`, which adds the current directory to Julia's `LOAD_PATH`.
    
    push!(LOAD_PATH, pwd())

The package also reads and writes binary VTK files natively using a small wrapper library written in C++ that yields `libvtk.so`. To build it, first edit the `src.cpp/CMakeLists.txt` to reflect where your *shared* VTK libraries (e.g.,`libvtkCommon.so`) are installed.

    include_directories(
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      /usr/include/vtk-5.x/    
      )
    
    link_directories(
      /usr/lib/
      )  

You may have to install VTK by hand to get the shared libraries built. If you install them somewhere non-standard (like shown in this example), you'll have to tell system where to find them. On an Ubuntu machine, create the file `/etc/ld.so.conf.d/vtk.conf` and add `/opt/lib/vtk-5.10/` to it. Then run `sudo ldconfig`, and this library will be added to the shared library search path.

To build our wrapper `libvtk.so`, it is recommended to do so "out of source". You will also need `cmake`.

    cd lib/vtk
    cmake ../../src.cpp

This will create `libvtk.so` as an artifact. The file `src.jl/TetFemConfig.jl` is the current way to configure runtime properties of the solver. The `vtklib_loc` constant tells where to look for the library. It defaults to `lib/vtk/libvtk.so`, so you just have to adjust the rest of the path to your machine.

## Usage

From within the `src.jl` directory, open the julia repl (`julia`) and type `import FlemMain`

    rietmann@wyoming ~/Dropbox/PostDoc/TetFemJulia/src.jl> julia
                   _
       _       _ _(_)_     |  A fresh approach to technical computing
      (_)     | (_) (_)    |  Documentation: http://docs.julialang.org
       _ _   _| |_  __ _   |  Type "?help" for help.
      | | | | | | |/ _` |  |
      | | |_| | | | (_| |  |  Version 0.4.1 (2015-11-08 10:33 UTC)
     _/ |\__'_|_|_|\__'_|  |  Official http://julialang.org release
    |__/                   |  x86_64-linux-gnu
    
    julia> import FlemMain
    INFO: Loading additional PyPlot commands for graphing for SymPy objects:
    ...Info about PyPlot and Warnings about the Match library...
    Loading /scratch/tetfem/2x2_mesh_1.vtk
    loading lib /home/rietmann/Dropbox/PostDoc/TetFemJulia/lib/vtk/libvtk.so
    ...More warnings about a library...
    Simulating for T=2.82842712474619 in 197 steps @ dt=0.014357498095158324
    |Error|_inf exact: 2.747102495459508e-5
    |Error|_inf exact: 3.287062794210538e-5

If you make a change, say `finaltime = 4*2/sqrt(2)` in `RunSimulation2d.jl`, you have to reload both modules in order:

    julia> reload("RunSimulation2d"); reload("FlemMain")
    WARNING: replacing module RunSimulation2d
    WARNING: replacing module FlemMain
    Loading /scratch/tetfem/2x2_mesh_1.vtk
    loading lib /home/rietmann/Dropbox/PostDoc/TetFemJulia/lib/vtk/libvtk.so
    Simulating for T=5.65685424949238 in 393 steps @ dt=0.014394031169191806
    |Error|_inf exact: 2.7652742896955296e-5
    |Error|_inf exact: 3.279227190144951e-5
    |Error|_inf exact: 2.9038929034275185e-5
    |Error|_inf exact: 3.0309290195096317e-5
    0-element Array{Any,1}
    
Julia's module-loading system is still a bit primitive, so if you make a change in `TimeStepping.jl`, you have to reload it, followed by `RunSimulation2d` because `TimeStepping` is loaded by `RunSimulation2d`. 

    julia> reload("TimeStepping"); reload("RunSimulation2d"); reload("FlemMain")

If you make a change and the flow of modules is too complicated, just exit julia (CTRL-d) and `reload("FlemMain")`.



