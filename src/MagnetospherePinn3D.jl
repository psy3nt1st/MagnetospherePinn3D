    module MagnetospherePinn3D

    using ComponentArrays
    using Dates
    using Distributions
    using DrWatson
    using JLD2
    using Lux
    using LuxCUDA
    using LineSearches
    using OptimizationOptimJL
    using OrderedCollections
    using PrettyPrint
    using Printf
    using ProgressMeter
    using Random
    using Zygote

    export 
        # Configuration function
        create_config,

        # File management
        setup_jobdir,
        setup_subjobdir,

        # Main function
        main,

        # DrWatson functions
        dict_list,

        # Neural network functions
        generate_input, 
        create_neural_network, 
        setup_optprob,
        train_pinn!,
        gpu_device,
        evaluate_subnetworks,
        loss_function,

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

        # Equations
        calculate_derivatives,
        calculate_divergence,
        calculate_Bdotgradα,
        calculate_r_equation,
        calculate_θ_equation,
        calculate_ϕ_equation

    include("Config.jl")
    include("Equations.jl")
    include("Trainning.jl")
    include("Utilities.jl")

    end # module MagnetospherePinn3D
