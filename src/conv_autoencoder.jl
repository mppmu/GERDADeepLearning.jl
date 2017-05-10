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
  X = mx.Reshape(X, shape=(1, input_size, 1, 0), name=:reshape_for_conv) # last argument is batch_size
  for i in 1:length(filter_counts)
    X = conv_layer("conv_$i", X, filter_counts[i], filter_lengths[i], act_type, pool_sizes[i], pool_type, conv_dropout)
  end
  X = mx.Flatten(X)

  # Fully connected layers
  for i in 1:(length(fc_layers)-1)
    X = fc_layer("fc_$i", X, fc_layers[i], act_type, dropout)
  end
  X = fc_layer("latent", X, fc_layers[end], act_type, dropout)
  return _build_conv_decoder(X, Y, properties, input_size)
end

function _build_conv_decoder(full_size, properties, input_size)
  X = mx.Variable(:data)
  Y = mx.Variable(:label)
  return _build_conv_decoder(X, Y, properties, full_size)
end

function _build_conv_decoder(X::mx.SymbolicNode, Y::mx.SymbolicNode, properties, full_size)
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

  last_conv_data_length = Int(full_size / prod(pool_sizes))

  for i in (length(fc_layers)-1):-1:1
    X = fc_layer("dec_fc_$i", X, fc_layers[i], act_type, dropout)
  end
  X = fc_layer("blow_up", X, filter_counts[end] * last_conv_data_length, act_type, conv_dropout)

  # Deconvolutions
  X = mx.Reshape(X, shape=(1, last_conv_data_length, Int(filter_counts[end]), 0), name=:reshape_for_deconv)
  for i in length(filter_counts):-1:2
    X = deconv_layer("deconv_$i", X, filter_counts[i-1], filter_lengths[i], act_type, pool_sizes[i], conv_dropout)
  end
  X = deconv_layer("deconv_1", X, 1, filter_lengths[1], nothing, pool_sizes[1], 0)
  # output should have shape (features, batch_size)
  X = mx.Flatten(X, name=:out) # (batch_size, width, height=1)
    # X = mx.FullyConnected(X, num_hidden=input_size, name=:out)
  loss = mx.LinearRegressionOutput(X, Y, name=:softmax)
  return loss, X
end


export autoencoder
function autoencoder(env::DLEnv,
    training_data::EventLibrary, eval_data::EventLibrary; id="autoencoder", action::Symbol=:auto)

  if action == :auto
    action = decide_best_action(network(env,id))
    info("$id: auto-selected action is $action")
  end

  n = network(env, id)

  train_provider = mx.ArrayDataProvider(:data => training_data.waveforms,
      :label => training_data.waveforms, batch_size=n["batch_size"])
  eval_provider = mx.ArrayDataProvider(:data => eval_data.waveforms,
      :label => eval_data.waveforms, batch_size=n["batch_size"])

  build(n, action, train_provider, eval_provider, _build_conv_autoencoder)
  return n
end

function autoencoder(env::DLEnv, data_sets::Dict{Symbol, EventLibrary};
  id="autoencoder", action::Symbol=:auto,
  train_key=:train, xval_key=:xval)
  return autoencoder(env, data_sets[train_key], data_sets[xval_key]; id=id, action=action)
end


export decoder
function decoder(env::DLEnv, latent_datasets::Dict{Symbol, EventLibrary},
  target_datasets::Dict{Symbol, EventLibrary}; id="decoder", action::Symbol=:auto, train_key=:train, xval_key=:xval)

  if action == :auto
    action = decide_best_action(network(env,id))
    info("$id: auto-selected action is $action")
  end

  n = network(env, id)

  train_provider = mx.ArrayDataProvider(:data => latent_datasets[train_key].waveforms,
      :label => target_datasets[train_key].waveforms, batch_size=n["batch_size"])
  eval_provider = mx.ArrayDataProvider(:data => latent_datasets[xval_key].waveforms,
      :label => target_datasets[xval_key].waveforms, batch_size=n["batch_size"])

  full_size = size(target_datasets[train_key].waveforms, 1)
  build(n, action, train_provider, eval_provider, (p, s) -> _build_conv_decoder(full_size, p, s))
  return n
end


export encode
function encode(events::EventLibrary, n::NetworkInfo)
  info("$(n.name): encoding '$(name(events))'...")
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

function encode(sets::Dict{Symbol,EventLibrary}, n::NetworkInfo)
  mapvalues(sets, encode, n)
end

export decode
function decode(compact::EventLibrary, n::NetworkInfo, pulse_size)
  info("$(n.name): decoding '$(name(compact))'...")

  X = mx.Variable(:data)
  Y = mx.Variable(:label) # not needed because no training
  loss, X = _build_conv_decoder(X, Y, n.config, pulse_size)
  model = subnetwork(n.model, mx.FeedForward(loss, context=best_device()))

  batch_size=n["batch_size"]
  if length(compact) < batch_size
      compact.waveforms = hcat(compact.waveforms, fill(0, size(compact.waveforms, 1), batch_size-size(compact.waveforms, 2)))
  end

  provider = mx.ArrayDataProvider(:data => compact.waveforms, batch_size=batch_size)
  transformed = mx.predict(model, provider)

  result = copy(compact)
  result.waveforms = transformed
  setname!(result, name(result)*"_decoded")
  push_classifier!(result, "Autoencoder")
  return result
end

function decode(sets::Dict{Symbol,EventLibrary}, n::NetworkInfo, pulse_size)
  mapvalues(sets, decode, n, pulse_size)
end

export mse
function mse(events::EventLibrary, n::NetworkInfo)
  provider = mx.ArrayDataProvider(:data => events.waveforms,
      :label => events.waveforms, batch_size=n["batch_size"])
  mse_result = eval(n.model, provider, mx.MSE())
  return mse_result[1][2]
end

function mse(libs::Dict{Symbol,EventLibrary}, n::NetworkInfo)
  mapvalues(libs, mse, n)
end

export encode_decode
function encode_decode(events::EventLibrary, n::NetworkInfo)
  batch_size=n["batch_size"]
  provider = padded_array_provider(:data, events.waveforms, batch_size)
  reconst = mx.predict(n.model, provider)

  result = copy(events)
  result.waveforms = reconst
  setname!(result, "$(name(result))_reconst")
  push_classifier!(result, "Autoencoder")
  return result
end

function encode_decode(libs::Dict{Symbol,EventLibrary}, n::NetworkInfo)
  mapvalues(libs, encode_decode, n)
end
