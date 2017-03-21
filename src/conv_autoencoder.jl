# This file is a part of GERDADeepLearning.jl, licensed under the MIT License (MIT).

using MXNet



function _build_conv_autoencoder(properties, input_size)
  batch_size = properties["batch_size"]
  filter_counts = properties["conv_filters"]
  filter_lengths = properties["conv_lengths"]
  assert(length(filter_counts) == length(filter_lengths))
  pool_sizes = properties["pool_size"]
  pool_type = properties["pool_type"]
  act_type = properties["activation"]
  conv_dropout = properties["conv_dropout"]

  fc_layers = properties["fc"]
  dropout = properties["dropout"]

  last_conv_data_length = Int(input_size / prod(pool_sizes))

  X = mx.Variable(:data)
  Y = mx.Variable(:label)

  # Convolutional layers
  X = mx.Reshape(data=X, shape=(1, input_size, 1, batch_size))
  for i in 1:length(filter_counts)
    X = conv_layer("conv_$i", X, filter_counts[i], filter_lengths[i], act_type, pool_sizes[i], pool_type, conv_dropout)
  end
  X = mx.Flatten(X)

  # Fully connected layers
  for i in 1:(length(fc_layers)-1)
    X = fc_layer("fc_$i", X, fc_layers[i], act_type, dropout)
  end
  X = fc_layer("latent", X, fc_layers[end], act_type, dropout)
  for i in (length(fc_layers)-1):-1:1
    X = fc_layer("fc_$i", X, fc_layers[i], act_type, dropout)
  end
  X = fc_layer("blow_up", X, filter_counts[end] * last_conv_data_length, act_type, conv_dropout)

  # Deconvolutions
  X = mx.Reshape(data=X, shape=(1, last_conv_data_length, Int(filter_counts[end]), batch_size))
  for i in length(filter_counts):-1:2
    X = deconv_layer("deconv_$i", X, filter_counts[i-1], filter_lengths[i], act_type, pool_sizes[i], conv_dropout)
  end
  X = deconv_layer("deconv_1", X, 1, filter_lengths[1], nothing, pool_sizes[1], 0)
  # output should have shape (features, batch_size)
  X = mx.Flatten(X, name=:out) # (batch_size, width, height=1)
    # X = mx.FullyConnected(data=X, num_hidden=input_size, name=:out)
  loss = mx.LinearRegressionOutput(data=X, label=Y, name=:softmax)
  return loss, X
end


export autoencoder
function autoencoder(build_type::Symbol, env::DLEnv,
    training_data::EventLibrary, eval_data::EventLibrary, id="autoencoder")

  n = network(env, id)

  train_provider = mx.ArrayDataProvider(:data => training_data.waveforms,
      :label => training_data.waveforms, batch_size=n["batch_size"])
  eval_provider = mx.ArrayDataProvider(:data => eval_data.waveforms,
      :label => eval_data.waveforms, batch_size=n["batch_size"])

  build(n, build_type, train_provider, eval_provider, _build_conv_autoencoder)
  return n
end

function autoencoder(env::DLEnv, training_data::EventLibrary, eval_data::EventLibrary, id="autoencoder")
  action = decide_best_action(network(env,id))
  println("Autoencoder: auto-selected action is $action")
  return autoencoder(action, env, training_data, eval_data, id)
end

export encode
function encode(env::DLEnv, events::EventLibrary, n::NetworkInfo)
  println("Encoding '$(name(events))'...")
  model = n.model
  model = subnetwork(model.arch, model.arg_params, model.aux_params, "latent", true)
  provider = mx.ArrayDataProvider(:data => events.waveforms, batch_size=n["batch_size"])
  transformed = mx.predict(model, provider)

  result = copy(events)
  result.waveforms = transformed
  setname!(result, name(result)*"_encoded")

  push_classifier!(result, "Autoencoder")

  return result
end

function encode(env::DLEnv, sets::Dict{Symbol,EventLibrary}, n::NetworkInfo, lib_name="latent")
  result = Dict{Symbol, EventLibrary}()
  for (key, value) in sets
    result[key] = encode(env, value, n)
  end

  push!(env, lib_name, result)

  return result
end
