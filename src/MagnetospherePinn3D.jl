module MagnetospherePinn3D

using ComponentArrays
using Configurations
using Dates
using DifferentialEquations
using Distributions
using Integrals
using JLD2
using LegendrePolynomials
using Lux
using LuxCUDA
using LineSearches
using OptimizationOptimJL
using Parameters
using Printf
using ProgressMeter
using Random
using TOML
using Zygote




# import OptimizationOptimisers: Adam


export 
       # Types
       Params,  
       
       # Packages
       Lux,  
       Random, 

       # Neural network functions
       generate_input, 
       create_neural_network, 
       setup_optprob,
       train_neural_network!,
       gpu_device,

       # Variables
       α,
       Br,
       Bθ,
       Bϕ,
       B_mag,

       # File management
       setup_jobdir,
       import_params, 
       export_params,

       # Equations
       grad,
       diver,
       curl,
       laplacian,
       scalar_product,
       calculate_derivatives,
       calculate_divergence,
       calculate_Bdotgradα,
       calculate_r_equation,
       calculate_θ_equation,
       calculate_ϕ_equation,
       calculate_divergence,
       calculate_Bdotgradα,
       calculate_gradB2,

       
       # PostProcess
       create_test,
       calculate_energy,
       find_footprints,
       integrate_fieldlines!,
       findnearest


       # plot_fieldlines      
       # Variables,
       # Parameters,
       # α_surface,
       # f_boundary, 
       # h_boundary, 
       # initialize,
       # loss_function,
       # Optimizers
       # Adam,
       # BFGS,
       # LBFGS,



include("TypesForceFree3D.jl")
include("FunctionsForceFree3D.jl")
include("PostProcessForceFree3D.jl")
# include("Utils.jl")



end # module MagnetospherePinn3D
