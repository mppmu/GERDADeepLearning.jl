# This file is a part of GERDADeepLearning.jl, licensed under the MIT License (MIT).

using MXNet

import Base: getindex


function fc_layer(name, X, n_hidden, act_type, pdropout)
  fc = mx.FullyConnected(data=X, num_hidden=Int(n_hidden), name="$(name)")
  act = mx.Activation(data=fc, act_type=act_type, name="$(name)_act")
  return mx.Dropout(data=act, p=pdropout, name="$(name)_dropout")
end

function conv_layer(name, X, num_filter, filter_length, act_type, pool_size, pool_type, dropout)
  conv = mx.Convolution(data=X, kernel=(1, filter_length), pad=(0, 4), num_filter=num_filter, name="$name") # 2D->3D
  act = mx.Activation(data=conv, act_type=act_type, name="$(name)_act")
  pool = mx.Pooling(data=act, kernel=(1, pool_size), stride=(1, pool_size), pool_type=pool_type, name="$(name)_pool") # 256->64
  dropout = mx.Dropout(data=pool, p=dropout, name="$(name)_dropout")
  return dropout
end

function deconv_layer(name, X, num_filter, filter_length, act_type, pool_size, dropout)
  pad = Int( floor(filter_length/2) - pool_size/2 )
  if pad < 0
    throw(ArgumentError("upsampling not possible because padding is negative ($pad). Increase filter length or decrease pool size."))
  end
  X = mx.Deconvolution(data=X, kernel=(1, filter_length-1), stride=(1, pool_size), pad=(0, pad), num_filter=num_filter, name="$name")
  if act_type != nothing
    X = mx.Activation(data=X, act_type=act_type, name="$(name)_act")
  end
  X = mx.Dropout(data=X, p=dropout, name="$(name)_dropout")
  return X
end



type PlotCallback <: mx.AbstractEpochCallback
    graphs::Dict{Symbol, Array{Float32,1}}

    PlotCallback() = new(Dict{Symbol, Array{Float32,1}}())
end

function (cb::PlotCallback){T<:Real}(model :: Any, state :: mx.OptimizationState, metric :: Vector{Tuple{Base.Symbol, T}})
  for index in 1:length(metric)
    key = metric[index][1]
    if !haskey(cb.graphs, key)
      info("PlotCallback: creating key $key")
      cb.graphs[key] = Float32[]
    end

    push!(cb.graphs[key], metric[index][2])
  end
end




function calculate_parameters(model, filepath)
  file = open(filepath, "w")
  total_parameter_count = 0
  for param in model.arg_params
    line = "Parameter $(param[1]) has shape $(size(param[2])) = $(length(param[2]))"
    println(line)
    write(file, line*"\n")
    total_parameter_count += length(param[2])
  end
  lastline = "Total parameter count: $total_parameter_count"
  println(lastline)
  write(file, lastline*"\n")
  close(file)
 end


export has_gpu
 function has_gpu()
   try
     mx.ones(Float32, (1,1), mx.gpu())
     return true
   catch err
     return false
   end
 end

 best_device_cache = nothing

 function best_device()
   global best_device_cache
   if best_device_cache == nothing
     best_device_cache = has_gpu() ? mx.gpu() : mx.cpu()
   end
   return best_device_cache
 end



 type NetworkInfo
   name::String
   dir::AbstractString
   config::Dict
   model
   epoch::Integer # the current state of the model, initialized to 0.

   NetworkInfo(name::String, dir::AbstractString, config::Dict) =
      new(name, dir, config, nothing, 0)
 end

export getindex
 function getindex(n::NetworkInfo, key)
  return n.config[key]
end


function train(n::NetworkInfo,
      train_provider, eval_provider,
      xpu)
  learning_rate = n["learning_rate"]
  epochs = n["epochs"]

  plot_cb = PlotCallback()

  metric = mx.MSE()

  optimizer = mx.ADAM(lr=learning_rate)
  println("Training on device $xpu")
  mx.fit(n.model, optimizer, train_provider,
         n_epoch=epochs-n.epoch,
         eval_metric=metric,
         callbacks=[plot_cb, mx.do_checkpoint(joinpath(n.dir,n.name))])

  # TODO eval not implemented

  n.epoch = epochs

  calculate_parameters(n.model, joinpath(n.dir,n.name*"-parameters.txt"))
  writedlm(joinpath(n.dir,n.name*"-mse.txt"), plot_cb.graphs[:MSE])
end


function build(n::NetworkInfo, method::Symbol,
    train_provider, eval_provider, build_function;
    xpu=best_device()
  )
  target_epoch = n["epochs"]
  slim = n["slim"]

  if(slim > 0)
    println("$(n.name): slim $(train_provider.sample_count) -> $slim")
    train_provider = slim_provider(train_provider, slim)
    if eval_provider != nothing
      eval_provider = slim_provider(eval_provider, slim)
    end
  end

  if method == :train
    loss, net = build_function(n.config, size(train_provider.data_arrays[1],1))
    n.model = mx.FeedForward(loss, context=xpu)
    train(n, train_provider, eval_provider, xpu)
  elseif method == :load
    load_network(n, target_epoch)
  elseif method == :refine
    load_network(n, -1)
    train(n, train_provider, eval_provider, xpu)
  else
    throw(ArgumentError("method must be train, load or refine. got $method"))
  end
end

function slim_provider(p::mx.ArrayDataProvider, slim)
  slim = min(slim, p.sample_count)
  return mx.ArrayDataProvider(p.data_names[1] => slim_array(p.data_arrays[1], slim),
      p.label_names[1] => slim_array(p.label_arrays[1], slim); batch_size=p.batch_size)
end

function slim_array(array, slim)
  if(length(size(array)) == 1)
    return array[1:slim]
  else
    return array[:,1:slim]
  end
end



 function load_network(n::NetworkInfo, epoch; output_name="softmax", delete_unneeded_arguments=true)
   if epoch < 0
     epoch = last_epoch(n.dir, n.name)
   end

   sym, arg_params, aux_params = mx.load_checkpoint(joinpath(n.dir, n.name), epoch)
   n.model = subnetwork(sym, arg_params, aux_params, output_name, delete_unneeded_arguments)
   n.epoch = epoch
   return n
 end

 function subnetwork(sym, arg_params, aux_params, output_name, delete_unneeded_arguments; xpu=best_device())
   all_layers = mx.get_internals(sym)
   loss = all_layers[output_name*"_output"]

   model = mx.FeedForward(loss, context=xpu)
   model.arg_params = copy(arg_params)
   model.aux_params = copy(aux_params)

   if delete_unneeded_arguments
     needed_args = mx.list_arguments(loss)
     for (name, array) in model.arg_params
       if !(name in needed_args)
         delete!(model.arg_params, name)
       end
     end
   end

   return model
 end

 function last_epoch(dir, prefix; start=1)
   i = start
   if !isfile("$dir/$prefix-$(lpad(i,4,0)).params")
     throw(ArgumentError("Model not found: $dir"))
   end

   while isfile("$dir/$prefix-$(lpad(i,4,0)).params")
     i += 1
   end
   return i-1
 end

 function exists_network(dir::AbstractString, name::AbstractString)
   return isfile("$dir/$name-symbol.json")
 end

function decide_best_action(n::NetworkInfo)
  if !exists_network(n.dir, n.name) return :train end
  if n["epochs"] > last_epoch(n.dir, n.name) return :refine end
  return :load
end
