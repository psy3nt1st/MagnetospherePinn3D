    module MagnetospherePinn3D

    using ComponentArrays
    using Configurations
    using Dates
    using Distributions
    using JLD2
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
        evaluate_subnetworks,
        loss_function,
        create_subnet,

        # Variables
        α,
        Br,
        Bθ,
        Bϕ,
        B_mag,

        # Boundary_conditions
        α_surface,
        Br_surface,
        h_boundary,

        # File management
        setup_jobdir,
        import_params, 
        export_params,
        setup_configfile,

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
        calculate_αS_equation

    include("Equations.jl")
    include("Trainning.jl")
    include("Types.jl")
    include("Utilities.jl")

    end # module MagnetospherePinn3D
