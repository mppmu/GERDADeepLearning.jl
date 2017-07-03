# This file is a part of GERDADeepLearning.jl, licensed under the MIT License (MIT).

using MXNet

import StatsBase: predict


function _build_dnn_classifier(properties, input_size)
  batch_size = properties["batch_size"]
  act_type = properties["activation"]
  fc_sizes = properties["fc"]
  dropout = properties["dropout"]

  X = mx.Variable(:data)
  Y = mx.Variable(:label)

  for i in 1:length(fc_sizes)
    X = fc_layer("fc_$i", X, fc_sizes[i], act_type, dropout)
  end
  X = mx.FullyConnected(X, num_hidden=1, name=:out)

  # Available outputs: SoftmaxOutput, LinearRegressionOutput, LogisticRegressionOutput, MAERegressionOutput, SVMOutput
  loss = mx.LogisticRegressionOutput(X, Y, name=:softmax)
  return loss, X
end



export dnn_classifier
function dnn_classifier(env::DLEnv, data::DLData;
  id="dnn-classifier", action::Symbol=:auto, label_key=:SSE,
  train_key="train", xval_key="xval", evaluate=["test"])

  if action == :auto
    action = decide_best_action(network(env,id))
    info(env, 2, "$id: auto-selected action is $action")
  end

  n = network(env, id)

  if action != :load
    training_data = flatten(data[:set=>train_key])
    xval_data = flatten(data[:set=>xval_key])

    # Preprocessing
    if !haskey(training_data, label_key)
      info(env, 2, "$id: No labels found on training data -> Using default energy labels from label_energy_peaks()")
      label_energy_peaks(training_data, label_key)
      label_energy_peaks(xval_data, label_key)
    end
    if length(find(x->x==-1, training_data[label_key])) > 0
      info(env, 2, "$id: Removing unlabeled events and equalizing class counts.")
      training_data, indices = equalize_counts_by_label(training_data, label_key)
      xval_data, indices = equalize_counts_by_label(xval_data, label_key)
    end


    if eventcount(xval_data) < n["batch_size"]
      n["batch_size"] = eventcount(xval_data)
      info("Cross validation set only has $(eventcount(xval_data)) data points. Adjusting bach size accordingly.")
    end

    train_provider = mx.ArrayDataProvider(:data => waveforms(training_data),
        :label => transpose(training_data[label_key]), batch_size=n["batch_size"])
    xval_provider = mx.ArrayDataProvider(:data => waveforms(xval_data),
        :label => transpose(xval_data[label_key]), batch_size=n["batch_size"])
  else
    train_provider = nothing
    xval_provider = nothing
  end

  min_eval_count = minimum([eventcount(data[:set=>e]) for e in evaluate])
  if min_eval_count < n["batch_size"]
    n["batch_size"] = min_eval_count
    info("A test set only has $min_eval_count data points. Adjusting bach size accordingly.")
  end

  build(n, action, train_provider, xval_provider, _build_dnn_classifier)

  for eval_set_name in evaluate
    for lib in data[:set=>eval_set_name]
      predict(lib, n)
    end
  end

  return n
end

export predict
function predict(data::EventLibrary, n::NetworkInfo, psd_name=:psd)
  if eventcount(data) > 0
    provider = mx.ArrayDataProvider(:data => waveforms(data), batch_size=n["batch_size"])
    predictions = mx.predict(n.model, provider)
    data.labels[psd_name] = predictions[1,:]
  else
    data.labels[psd_name] = zeros(Float32, 0)
  end
  push_classifier!(data, "Neural Network")
end

predict(data::DLData, n::NetworkInfo; psd_name=:psd) = mapvalues(data, predict, n, psd_name)
