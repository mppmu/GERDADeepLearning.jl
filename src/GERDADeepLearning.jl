# This file is a part of GERDADeepLearning.jl, licensed under the MIT License (MIT).

# __precompile__(false)

module GERDADeepLearning

include("gdl_util.jl")
include("gerda_files.jl")

include("events.jl")
include("hdf5io.jl")
include("rootio.jl")
include("dl_env.jl")

include("signal_processing.jl")

include("mx_train_util.jl")
include("conv_autoencoder.jl")
include("dnn_classifier.jl")

include("efficiencies.jl")
include("plotting.jl")

end # module
