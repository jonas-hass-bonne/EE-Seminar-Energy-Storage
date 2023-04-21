# EE-Seminar-Energy-Storage
This repository contains data and code used in the model simulations for the seminar paper on the role of lithium-ion batteries in the Danish power system.

## Running the simulations
In order to run the model simulation, a working copy of Julia must be installed, which can be obtained by [following this link](https://julialang.org/downloads/).

Additionally, the following Julia packages must be installed:
- JuMP
- HiGHS
- Plots
- StatsPlots
- DataFrames
- CSV
- XLSX

To enable faster run times and easy editing of code files, the following packages are also recommended to work with the project although they are not strictly required:
- Revise
- PackageCompiler

These packages can be installed from the Julia REPL using the following command:
    using Pkg
    Pkg.add(["JuMP", "HiGHS", "Plots", "StatsPlots", "DataFrames", "CSV", "XLSX", "Revise", "PackageCompiler"])

The project can the be run be including the main.jl file in the REPL.
To speed up initial compile-time, the create-sysimage.jl file can be run to build a precompiled system image, which can then be used when starting the Julia REPL by specifying the --sysimage path/to/sys_project.so launch flag
Additionally, to speed up scenario simulations the Julia REPL can be started with access to multiple threads by specfiying the --threads #N_threads launch flag where #N_threads is the number of threads the Julia REPL will have access to.

