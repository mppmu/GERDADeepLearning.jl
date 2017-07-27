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

  # last_conv_data_length = Int(input_size / prod(pool_sizes))

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
  pool_sizes = convert(Vector{Int64}, properties["pool_size"])
  pool_type = properties["pool_type"]
  act_type = properties["activation"]
  conv_dropout = properties["conv_dropout"]
  fc_layers = properties["fc"]
  dropout = properties["dropout"]

  last_conv_data_length = Int(full_size / prod(pool_sizes))

  for i in (length(fc_layers)-1):-1:1
    X = fc_layer("dec_fc_$i", X, fc_layers[i], act_type, dropout)
  end
  if length(filter_counts) > 0
    X = fc_layer("blow_up", X, filter_counts[end] * last_conv_data_length, act_type, conv_dropout)
    X = mx.Reshape(X, shape=(1, last_conv_data_length, Int(filter_counts[end]), 0), name=:reshape_for_deconv)
  else
    X = fc_layer("blow_up", X, full_size, act_type, conv_dropout)
  end

  # Deconvolutions
  for i in length(filter_counts):-1:2
    X = deconv_layer("deconv_$i", X, filter_counts[i-1], filter_lengths[i], act_type, pool_sizes[i], conv_dropout)
  end
  if length(filter_counts) > 0
    X = deconv_layer("deconv_1", X, 1, filter_lengths[1], nothing, pool_sizes[1], 0)
    # output should have shape (features, batch_size)
    X = mx.Flatten(X, name=:out) # (batch_size, width, height=1)
  end
    # X = mx.FullyConnected(X, num_hidden=input_size, name=:out)
  loss = mx.LinearRegressionOutput(X, Y, name=:softmax)
  return loss, X
end


export autoencoder
function autoencoder(env::DLEnv, data::EventCollection; id="autoencoder", action::Symbol=:auto, train_key="train", xval_key="xval")

  if action == :auto
    action = decide_best_action(network(env,id))
    info(env, 2, "$id: auto-selected action is $action")
  end

  n = network(env, id)

  if action != :load
    training_waveforms = waveforms(data[:set=>train_key])
    xval_data = data[:set=>xval_key]

    if eventcount(xval_data) < n["batch_size"]
      n["batch_size"] = eventcount(xval_data)
      info("Cross validation set only has $(eventcount(xval_data)) data points. Adjusting bach size accordingly.")
    end
    xval_waveforms = waveforms(xval_data)
    train_provider = mx.ArrayDataProvider(:data => training_waveforms,
        :label => training_waveforms, batch_size=n["batch_size"])
    eval_provider = mx.ArrayDataProvider(:data => xval_waveforms,
        :label => xval_waveforms, batch_size=n["batch_size"])
  else
    train_provider = nothing
    eval_provider = nothing
  end

  build(n, action, train_provider, eval_provider, _build_conv_autoencoder; verbosity=get_verbosity(env))
  return n
end


export decoder
function decoder(env::DLEnv, latent_data::EventCollection,
  target_data::EventCollection; id="decoder", action::Symbol=:auto, train_key=:train, xval_key=:xval)

  if action == :auto
    action = decide_best_action(network(env,id))
    info(env, 2, "$id: auto-selected action is $action")
  end

  n = network(env, id)

  train_provider = mx.ArrayDataProvider(:data => waveforms(latent_data[:set=>train_key]),
      :label => waveforms(target_data[:set=>train_key]), batch_size=n["batch_size"])
  eval_provider = mx.ArrayDataProvider(:data => waveforms(latent_data[:set=>xval_key]),
      :label => waveforms(target_data[:set=>xval_key]), batch_size=n["batch_size"])

  full_size = sample_size(target_data)
  build(n, action, train_provider, eval_provider, (p, s) -> _build_conv_decoder(full_size, p, s);
  verbosity=get_verbosity(env))
  return n
end


export encode
function encode(events::EventLibrary, n::NetworkInfo; log=false)
  log && info("$(n.name): encoding '$(name(events))'...")
  model = n.model
  model = subnetwork(model.arch, model.arg_params, model.aux_params, "latent", true, n.context)

  result = copy(events)

  if eventcount(events) > 0
        println(eventcount(events))
        println(size(waveforms(events),2))
    provider = mx.ArrayDataProvider(:data => waveforms(events), batch_size=min(n["batch_size"], eventcount(events)))
    transformed = mx.predict(model, provider)
    result.waveforms = transformed
  else
    result.waveforms = zeros(Float32, 0, 0)
  end

  setname!(result, name(result)*"_encoded")
  push_classifier!(result, "Autoencoder")
  return result
end

function encode(data::DLData, n::NetworkInfo)
  mapvalues(data, encode, n)
end

export decode
function decode(compact::EventLibrary, n::NetworkInfo, pulse_size; log=false)
  log && info("$(n.name): decoding '$(name(compact))'...")

  X = mx.Variable(:data)
  Y = mx.Variable(:label) # not needed because no training
  loss, X = _build_conv_decoder(X, Y, n.config, pulse_size)
  model = subnetwork(n.model, mx.FeedForward(loss, context=n.context))

  batch_size=n["batch_size"]
  if eventcount(compact) < batch_size
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

function decode(data::DLData, n::NetworkInfo, pulse_size)
  mapvalues(data, decode, n, pulse_size)
end

export mse
function mse(events::EventLibrary, n::NetworkInfo)
  provider = mx.ArrayDataProvider(:data => events.waveforms,
      :label => events.waveforms, batch_size=n["batch_size"])
  mse_result = eval(n.model, provider, mx.MSE())
  return mse_result[1][2]
end

function mse(data::DLData, n::NetworkInfo)
  mapvalues(data, mse, n)
end

export encode_decode
function encode_decode(events::EventLibrary, n::NetworkInfo)
  batch_size=n["batch_size"]
  provider = padded_array_provider(:data, waveforms(events), batch_size)
  reconst = mx.predict(n.model, provider)

  result = copy(events)
  result.waveforms = reconst
  setname!(result, "$(name(result))_reconst")
  push_classifier!(result, "Autoencoder")
  return result
end

function encode_decode(data::DLData, n::NetworkInfo)
  mapvalues(data, encode_decode, n)
end
