module Config

# Define the configuration dictionary
const config = Dict(
    :learning_rate => 0.001,
    :batch_size => 32,
    :epochs => 100,
    :activation_function => "relu",
    :optimizer => "adam",
    :input_dim => 3,
    :output_dim => 1,
    :hidden_layers => [64, 64, 64],
    :regularization => 1e-4
)

end # module Config