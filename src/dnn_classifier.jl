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
  X = mx.FullyConnected(data=X, num_hidden=1, name=:out)

  # Available outputs: SoftmaxOutput, LinearRegressionOutput, LogisticRegressionOutput, MAERegressionOutput, SVMOutput
  loss = mx.LogisticRegressionOutput(data=X, label=Y, name=:softmax)
  return loss, X
end



export dnn_classifier
function dnn_classifier(env::DLEnv, data_sets;
  id="dnn-classifier", action::Symbol=:auto, label_key=:SSE,
  train_key=:train, xval_key=:xval, evaluate=[:test])
  if action == :auto
    action = decide_best_action(network(env,id))
    println("$id: auto-selected action is $action")
  end

  training_data = data_sets[train_key]
  xval_data = data_sets[xval_key]

  # Preprocessing
  if !haskey(training_data, label_key)
    println("$id: preprocessing - adding default labels and equalizing class count...")
    training_data, indices = equalize_counts_by_label(training_data, label_key)
    xval_data, indices = equalize_counts_by_label(xval_data, label_key)
  end

  n = network(env, id)

  train_provider = mx.ArrayDataProvider(:data => training_data.waveforms,
      :label => training_data[label_key], batch_size=n["batch_size"])
  xval_provider = mx.ArrayDataProvider(:data => xval_data.waveforms,
      :label => xval_data[label_key], batch_size=n["batch_size"])

  build(n, action, train_provider, xval_provider, _build_dnn_classifier)

  for eval_set_name in evaluate
    eval_set = data_sets[eval_set_name]
    predict(env, eval_set, n)
  end

  return n
end

export predict
function predict(env::DLEnv, data::EventLibrary, n::NetworkInfo; psd_name=:psd)
  provider = mx.ArrayDataProvider(:data => data.waveforms, batch_size=n["batch_size"])
  predictions = mx.predict(n.model, provider)
  data.labels[psd_name] = predictions[1,:]
  push_classifier!(data, "Neural Network")
end
