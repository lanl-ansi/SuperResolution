# Scientific Data Compression and Super-Resolution Sampling


Illustrative examples for data compression and super-resolution sampling 


## Installation
The Julia package manager can be used to install the required packages as follows, 
```
] add GraphicalModelLearning
] add LinearAlgebra
] add StatsBase
] add JSON
] add FFTW
] add Distributions
] add DataFrames
] add JuMP
] add Ipopt
] add Plots
] add Graphs
] add GraphPlot

```

## Quick Start
1. Start Julia in your terminal:

```bash
julia
```

2. Once in the REPL (you’ll see the `julia>` prompt), include and run the desired example file:

```
include("Ising.jl")
```
This script demonstrates data compression and super-resolution sampling on the Ising model using varying DCT compression levels.
It outputs reconstructed results and generates comparative plots.


Alternatively, you can run the script directly from the command line:

```bash
julia Ising.jl
```

## License

This code is provided under a BSD license as part of the Optimization, Inference and Learning for Advanced Networks project, C18014.
