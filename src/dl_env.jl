# This file is a part of GERDADeepLearning.jl, licensed under the MIT License (MIT).

using JSON, Base.Threads, MultiThreadingTools

import Base: get, contains

export DLEnv
type DLEnv # immutable
  dir::AbstractString
  config::Dict
  _gpus::Vector{Int}
  _ext_event_libs::Dict{String,DLData}
  _verbosity::Integer # 0: nothing, 1: only errors, 2: default, 3: all

  DLEnv() = DLEnv("config")
  DLEnv(name::AbstractString) = DLEnv(abspath(""), name)
  function DLEnv(dir::AbstractString, name::AbstractString)
    f = open(joinpath(dir,"$name.json"), "r")
    dicttxt = readstring(f)  # file information to string
    dict = JSON.parse(dicttxt)  # parse and transform data
    env = new(dir, dict, Int[], Dict(), get(dict, "verbosity", 2))
    setup(env)
    return env
  end
end

type ConfigurationException <: Exception end

export get_properties
function get_properties(env::DLEnv, name::AbstractString)
  return get(env.config, name, Dict())
end

export set_properties!
function set_properties!(env::DLEnv, name::AbstractString, props::Dict)
  env.config[name] = props
end

export new_properties!
function new_properties!(modifier, env::DLEnv, template_name::AbstractString, new_name::AbstractString)
  d = copy(get_properties(env, template_name))
  modifier(d)
  set_properties!(env, new_name, d)
  return d
end

export resolvepath
function resolvepath(env::DLEnv, path::AbstractString...)
  if isabspath(path[1])
    return joinpath(path...)
  else
    return joinpath(env.dir, path...)
  end
end

function Base.joinpath(env::DLEnv, elements::String...)
  return joinpath(env.dir, elements...)
end

function Base.info(env::DLEnv, level::Integer, msg::AbstractString)
  if env._verbosity >= level
    threadsafe_info(msg)
  end
end

export set_verbosity!
function set_verbosity!(env::DLEnv, verbosity::Integer)
  env._verbosity = verbosity
end

export get_verbosity
function get_verbosity(env::DLEnv)
  return env._verbosity
end

function Base.getindex(env::DLEnv, key::String)
  return env.config[key]
end

function Base.setindex!(env::DLEnv, key::String, value)
  env.config[key] = value
end

function Base.haskey(env::DLEnv, key::String)
  return haskey(env.config, key)
end


export detectors
function detectors(env::DLEnv, dettype::AbstractString)
  if dettype == "BEGe"
    return BEGes_GERDA_II()
  elseif dettype == "coax" || dettype == "semi-coaxial"
    return Coax_GERDA_II()
  elseif dettype == "natural"
    return Natural_GERDA_II()
  elseif dettype == "used"
    return Used_GERDA_II()
  else
    throw(ArgumentError("Unknown detector type: $dettype"))
  end
end

function detectors(env::DLEnv, keywords::AbstractString...)
    sets = [detectors(env, keyword) for keyword in keywords]
    return intersect(sets...)
end

function detectors(env::DLEnv)
  return phase2_detectors
end


export _create_h5data
function _create_h5data(env::DLEnv, output_dir)
    info(env, 2, "Reading original data from $(env.config["path"])")
    isdir(output_dir) || mkdir(output_dir)
    if haskey(env.config, "rawformat")
        _seg_to_hdf5(env, output_dir)
    else
        _mgdo_to_hdf5(env, output_dir)
    end
    info(env, 3, "Converted raw data, HDF5 stored in $output_dir.")
end

function _mgdo_to_hdf5(env::DLEnv, output_dir)
  keylists = KeyList[]
  for (i,keylist_path) in enumerate(env.config["keylists"])
    if !endswith(keylist_path, ".txt")
      keylist_path = keylist_path*".txt"
    end
    push!(keylists, parse_keylist(resolvepath(env, keylist_path), keylist_path))
  end
  mgdo_to_hdf5(env.config["path"], output_dir, keylists; verbosity=get_verbosity(env))
end

function _seg_to_hdf5(env::DLEnv, output_dir)
    src_dirs = env.config["path"]
    all_files = []
    for src_dir in src_dirs
        content = readdir(src_dir)
        content = src_dir .* "/" .* content[find(f->endswith(f, ".root"), content)]
        append!(all_files, content)
    end
    seg_to_hdf5(env.config["rawformat"], all_files, output_dir, get_verbosity(env))
end

export getdata
function getdata(env::DLEnv; preprocessing::Union{AbstractString,Void}=nothing, targets::Array{String}=String[])
  if preprocessing==nothing
    return get(env, "raw"; targets=targets) do
      _get_raw_data(env; targets=targets)
    end
  else
    data = _get_raw_data(env; targets=[preprocessing])
    preprocessed = get(env, preprocessing; targets=targets) do
      preprocess(env, data, preprocessing)
    end
    if preprocessed == nothing
      return nothing
    end
    # Else check whether cache is up to date
    steps = get_properties(env, preprocessing)["preprocessing"]
    cache_up_to_date = true
    for lib in preprocessed
      cached_steps = lib[:preprocessing]
      if cached_steps != steps
        cache_up_to_date = false
      end
    end
    if cache_up_to_date
      return preprocessed
    else
      info(env, 2, "Refreshing cache of 'preprocessed'.")
      delete!(env, preprocessing)
      return getdata(env; preprocessing=preprocessing, targets=targets)
    end
  end
end

function _get_raw_data(env::DLEnv; targets::Array{String}=String[])
  raw_dir = resolvepath(env, "data", "raw")
  if !isdir(raw_dir)
    _create_h5data(env, raw_dir)
  end
  return lazy_read_all(raw_dir)
end


export preprocess
function preprocess(env::DLEnv, data::DLData, config_name)
  config = env.config[config_name]
  @assert isa(config, Dict)

  select_channels=parse_detectors(config["detectors"])
  info(env, 3, "Selected channels: $select_channels")
  if isa(select_channels,Vector) && length(select_channels) > 0
    filter!(data, :detector_name, select_channels)
  end

  _builtin_filter(env, config, "test-pulses", data, :isTP, isTP -> isTP == 0)
  _builtin_filter(env, config, "baseline-events", data, :isBL, isBL -> isBL == 0)
  _builtin_filter(env, config, "unphysical-events", data, :E, E -> (E > 0) && (E < 9999))
  _builtin_filter(env, config, "low-energy-events", data, :E, E -> (E > 0) && (E < 1000))

  N_dset = length(parse_datasets(env, config["sets"]))
  result = DLData(fill(EventLibrary(zeros(0,0)), length(data)*N_dset))
  result.dir = _cachedir(env, "tmp-preprocessed-$config_name")

  steps = convert(Array{String}, config["preprocessing"])

  for i in 1:length(data)
    lib = data.entries[i]
    info(env,2, "Preprocessing $(name(lib))")
    lib_t = preprocess_transform(env, lib, steps; copyf=identity)
    _builtin_filter(env, config, "failed-preprocessing", lib_t, :FailedPreprocessing, fail -> fail == 0)
    part_data = DLData(collect(values(split(env, lib_t, config["sets"]))))
    write_all_sequentially(part_data, result.dir, true)
    info(env,3, "Wrote datasets of $(name(lib)) and released allocated memory.")
    dispose(lib)
    dispose(lib_t)
    for j in 1:length(part_data)
      result.entries[N_dset*(i-1) + j] = part_data.entries[j]
    end
    @assert length(lib.waveforms) == 0 && length(lib_t.waveforms) == 0
  end

  return result
end


function _builtin_filter(env::DLEnv, config::Dict, ftype_key::String, data::EventCollection, label, exclude_prededicate)
  # TODO per detector to save memory
  ftype = config[ftype_key]
  info(env, 3, "Filter $ftype_key is set to $ftype.")
  if ftype == "exclude"
    result = filter!(data, label, exclude_prededicate)
    return result
  elseif ftype == "include"
    return data
  elseif ftype == "only"
    return filter!(data, label, x -> !exclude_prededicate(x))
  else
    throw(ArgumentError("Unknown filter keyword in configuration $ftype_key: $ftype"))
  end
end


function parse_datasets(env::DLEnv, strdict::Dict)
  result = Dict{AbstractString,Vector{AbstractFloat}}()
  requiredlength = length(env["keylists"])
  for (key,value) in strdict
    if isa(value, Vector)
      @assert length(value) == requiredlength
      result[key] = value
    elseif isa(value, AbstractFloat)
      result[key] = fill(value, requiredlength)
    else
      throw(ConfigurationException())
    end
  end
  return result
end

Base.split(env::DLEnv, data::EventCollection, strdict::Dict) = split(data, parse_datasets(env, strdict))


function setup(env::DLEnv)
  e_mkdir(env.dir)
  e_mkdir(joinpath(env.dir, "data"))
  e_mkdir(joinpath(env.dir, "models"))
  e_mkdir(joinpath(env.dir, "plots"))
end
e_mkdir(dir) = !isdir(dir) && mkdir(dir)

export get
function get(compute, env::DLEnv, lib_name::String; targets::Array{String}=String[], uninitialize=true)
  if !isempty(targets) && containsall(env, targets)
    info(env, 2, "Skipping retrieval of '$lib_name'.")
    return nothing
  end
  if contains(env, lib_name)
    info(env, 2, "Retrieving '$lib_name' from cache.")
    return get(env, lib_name)
  else
    info(env, 2, "Computing '$lib_name'...")
    data = compute()
    info(env, 3, "Computation of '$lib_name' finished.")

    # check type
    if !isa(data, DLData)
      throw(TypeError(Symbol(compute), "get_or_compute", DLData, data))
    end

    if data.dir == nothing
      info(env, 3, "Writing computed data to env as '$lib_name' (uninitialize=$uninitialize).")
      push!(env, lib_name, data; uninitialize=uninitialize)
      return data
    else
      # Rename directory
      targetdir = _cachedir(env, lib_name)
          if targetdir != data.dir
              info(env, 3, "Moving data from $(data.dir) to $targetdir and reloading.")
              mv(data.dir, targetdir)
              return get(env, lib_name)
            else
                info(env, 3, "Data is already in the right location ($(data.dir)).")
                return data
            end
    end
  end
end

function get(env::DLEnv, lib_name::String)
  _ensure_ext_loaded(env, lib_name)
  return env._ext_event_libs[lib_name]
end

Base.push!(env::DLEnv, lib_name::String, lib::EventLibrary; uninitialize::Bool=false) = push!(env, lib_name, DLData([lib]); uninitialize=uninitialize)

function Base.push!(env::DLEnv, lib_name::String, data::DLData; uninitialize::Bool=false)
  _ensure_ext_loaded(env, lib_name)
  env._ext_event_libs[lib_name] = data
  if env.config["cache"]
    write_all_multithreaded(data, joinpath(env.dir, "data", lib_name), uninitialize)
  end
end

export contains
function contains(env::DLEnv, lib_name::String)
  if haskey(env._ext_event_libs, lib_name)
    return true
  end
  return env.config["cache"] && isdir(_cachedir(env, lib_name))
end

export containsall
function containsall(env::DLEnv, lib_names::Array{String})
  for lib_name in lib_names
    if !contains(env, lib_name) return false end
  end
  return true
end

function _ensure_ext_loaded(env::DLEnv, lib_name::String)
  if !haskey(env._ext_event_libs, lib_name)
    c_dir = _cachedir(env, lib_name)
    if isdir(c_dir)
      env._ext_event_libs[lib_name] = lazy_read_all(c_dir)
    end
  end
end

function Base.delete!(env::DLEnv, lib_name::String)
  if haskey(env._ext_event_libs, lib_name)
    delete!(env._ext_event_libs, lib_name)
  end
  # Delete cached files (even if cache is set to false)
  c_dir = _cachedir(env, lib_name)
  if isdir(c_dir)
    rm(c_dir; recursive=true)
    info(env, 2, "Deleted cached $lib_name")
  end
end

function _cachedir(env::DLEnv, lib_name::String)
  return joinpath(env.dir, "data", lib_name)
end

function network(env::DLEnv, name::String)
    dir = joinpath(env.dir, "models", name)
    isdir(dir) || mkdir(dir)
    return NetworkInfo(name, dir, get_properties(env, name), to_context(env._gpus))
end

export use_gpus
function use_gpus(env, gpus::Int...)
  env._gpus = [gpus...]
end
