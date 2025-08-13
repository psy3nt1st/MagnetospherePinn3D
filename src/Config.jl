function create_config()
    
    return OrderedDict(
    # ---------------------------------------------------------------------------- #
    #                Parameters for the neural network architecture                #
    # ---------------------------------------------------------------------------- #
    :N_input => 4,
    :N_layers => 3,
    :N_neurons => 40,
    :N_output => 4,
    :N_points => 80000,
    :q_distribution => Uniform(0, 1),        # Uniform(0, 1) or Beta(3, 1)
    :rng_seed => nothing,                    # Integer or 'nothing' for random seed 
    # ---------------------------------------------------------------------------- #
    #                    Parameters for the optimization process                   #
    # ---------------------------------------------------------------------------- #
    :N_sets => 20,
    :adam_sets => 2,
    :adam_iters => 500,
    :quasiNewton_method => ["SSBroyden", "SSBFGS", "BFGS"],      # "SSBroyden", "SSBFGS", "BFGS"
    :quasiNewton_iters => 500,
    :linesearch => [HagerZhang(), MoreThuente(), BackTracking(), StrongWolfe()],           # HagerZhang() or MoreThuente() or BackTracking() or StrongWolfe()
    :loss_g => [identity, log],                     # identity or log. Function to apply to the MSE loss function
    :loss_normalization => "B",              # "q", "B", "none"
    :initialize_weights_from_previous => false,
    :initialize_weights_from_file => false,
    # ---------------------------------------------------------------------------- #
    #              Parameters for the physical properties of the model             #
    # ---------------------------------------------------------------------------- #
    :α_bc_mode => "hotspot",                # "hotspot", "double-hotspot", "axisymmetric"
    :multipole_bl => [[1.0, 0.0, 0.0]],     # Coefficients of the magnetic multipoles
    :compactness => 0.17,                   # Stellar compactness M/R. compactness = 0.0 for newtonian limit.
    :α0 => @onlyif(:α_bc_mode == "hotspot", 1.5),
    :θ1 => @onlyif(:α_bc_mode == "hotspot", 45.0),
    :ϕ1 => @onlyif(:α_bc_mode == "hotspot", 180.0),
    :σ  => @onlyif(:α_bc_mode == "hotspot", 0.2),
    :α0_b => @onlyif(:α_bc_mode == "double_hotspot", 1.5),
    :θ1_b => @onlyif(:α_bc_mode == "double_hotspot", 45.0),
    :ϕ1_b => @onlyif(:α_bc_mode == "double_hotspot", 120.0),
    :σ_b => @onlyif(:α_bc_mode == "double_hotspot",  0.2),
    :axisym_notation => @onlyif(:α_bc_mode == "axisymmetric", "Pc_n_γ"),           # "Pc_σ_s", "Pc_γ_n", "rc_σ_s"
    :Pc => @onlyif(:α_bc_mode == "axisymmetric", 0.0),
    :σ_gs => @onlyif(:α_bc_mode == "axisymmetric", 4.0),
    :s => @onlyif(:α_bc_mode == "axisymmetric", 0.0),
    :n => @onlyif(:α_bc_mode == "axisymmetric", 7),
    :γ => @onlyif(:α_bc_mode == "axisymmetric", 0.045),
    :rc => @onlyif(:α_bc_mode == "axisymmetric", 3.6),
    # ---------------------------------------------------------------------------- #
    #                            Other parameters                                  #
    # ---------------------------------------------------------------------------- #
    :jobdir => nothing
)

end