# This file is a part of GERDADeepLearning.jl, licensed under the MIT License (MIT).

using JSON

import Base: get, contains, push!

export DLEnv
type DLEnv # immutable
  dir::AbstractString
  config::Dict
  _ext_event_libs::Dict{String,Dict{Symbol,EventLibrary}}
  _verbosity::Integer # 0: nothing, 1: only errors, 2: default, 3: all

  DLEnv() = DLEnv("config")
  DLEnv(name::AbstractString) = DLEnv(abspath(""), name)
  function DLEnv(dir::AbstractString, name::AbstractString)
    f = open(joinpath(dir,"$name.json"), "r")
    dicttxt = readstring(f)  # file information to string
    dict = JSON.parse(dicttxt)  # parse and transform data
    env = new(dir, dict, Dict(), get(dict, "verbosity", 2))
    setup(env)
    return env
  end
end

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
  d = modifier(d)
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
    info(msg)
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


export _create_h5data
function _create_h5data(env::DLEnv)
  info(env, 2, "Reading original data from $(env.config["path"])")
  setdict = _setdict(env)
  set_names = collect(keys(setdict))
  keylists = KeyList[]
  for (i,keylist_path) in enumerate(env.config["keylists"])
    if !endswith(keylist_path, ".txt")
      keylist_path = keylist_path*".txt"
    end
    push!(keylists, parse_keylist(resolvepath(env, keylist_path), keylist_path))
  end
  raw_dir = resolvepath(env, "data", "raw")
  isdir(raw_dir) || mkdir(raw_dir)
  mgdo_to_hdf5(env.config["path"], raw_dir, keylists; verbosity=get_verbosity(env))
end

export getdata
function getdata(env::DLEnv; preprocessed=false, targets::Array{String}=String[])
  if !preprocessed
    return _get_raw_data(env; targets=targets)
  else
    data = _get_raw_data(env; targets=["preprocessed"])
    preprocessed = get(env, "preprocessed"; targets=targets) do
      preprocess(env, data)
    end
    if preprocessed == nothing
      return nothing
    end
    # Else check whether cache is up to date
    steps = env.config["preprocessing"]
    cache_up_to_date = true
    for (key,events) in preprocessed
      cached_steps = events[:preprocessing]
      if cached_steps != steps
        cache_up_to_date = false
      end
    end
    if cache_up_to_date
      return preprocessed
    else
      info(env, 2, "Refreshing cache of 'preprocessed'.")
      delete!(env, "preprocessed")
      return getdata(env; preprocessed=true, targets=targets)
    end
  end
end

function _get_raw_data(env::DLEnv; targets::Array{String}=String[])
  data = get(env, "data"; targets=targets) do
    # Else read original data
    info(env, 2, "Reading original data from $(env.config["path"])")
    keylist = FileKey[]
    setdict = _setdict(env)
    set_names = collect(keys(setdict))
    set_sizes = []
    for (i,keylist_path) in enumerate(env.config["keylists"])
      if !endswith(keylist_path, ".txt")
        keylist_path = keylist_path*".txt"
      end
      new_keys = parse_keylist(resolvepath(env, keylist_path))
      push!(keylist, new_keys...)
      push!(set_sizes, fill([setdict[setname][i] for setname in set_names], length(new_keys))...)
    end
    sets = read_tiers_1_4(env.config["path"], keylist,
        set_names=set_names, set_sizes=set_sizes,
        select_channels=parse_detectors(env.config["detectors"]),
        log_progress=env._verbosity >= 2, log_errors=env._verbosity >= 1)
    sets = _builtin_filter(env, "test-pulses", sets, :isTP, isTP -> isTP == 0)
    sets = _builtin_filter(env, "baseline-events", sets, :isBL, isBL -> isBL == 0)
    sets = _builtin_filter(env, "unphysical-events", sets, :E, E -> (E > 0) && (E < 9999))
    return sets
  end

  if data != nothing
    for (n, set) in data info(env, 2, summary(set)) end
  end
  return data
end




function _builtin_filter(env::DLEnv, ftype_key::String, sets::Dict{Symbol, EventLibrary}, label, exclude_prededicate)
  ftype = env.config[ftype_key]
  if ftype == "exclude"
    before = totallength(sets)
    result = filter(sets, label, exclude_prededicate)
    info(env, 2, "Excluded $(before-(totallength(result))) $ftype_key of $before events.")
    return result
  elseif ftype == "include"
    return sets
  elseif ftype == "only"
    return filter(sets, label, x -> !exclude_prededicate(x))
  else
    throw(ArgumentError("Unknown filter keyword in configuration $ftype_key: $ftype"))
  end
end


function _setdict(env::DLEnv)
  result = Dict{Symbol,Any}()
  strdict = get_properties(env, "sets")
  for (key,value) in strdict
    result[Symbol(key)] = value
  end
  return result
end

export setup
function setup(env::DLEnv)
  e_mkdir(env.dir)
  e_mkdir(joinpath(env.dir, "data"))
  e_mkdir(joinpath(env.dir, "models"))
  e_mkdir(joinpath(env.dir, "plots"))
end

function e_mkdir(dir)
  !isdir(dir) && mkdir(dir)
end

export get
function get(compute, env::DLEnv, lib_name::String; targets::Array{String}=String[])
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

    # check type
    if !isa(data, Dict)
      throw(TypeError(Symbol(compute), "get_or_compute", Dict{Symbol,EventLibrary}, data))
    end

    push!(env, lib_name, data)
    return data
  end
end

function get(env::DLEnv, lib_name::String)
  _ensure_ext_loaded(env, lib_name)
  return env._ext_event_libs[lib_name]
end

export push!
function push!(env::DLEnv, lib_name::String, libs::Dict{Symbol,EventLibrary})
  _ensure_ext_loaded(env, lib_name)
  dict = env._ext_event_libs[lib_name]
  for (key, val) in libs
    dict[key] = val
  end
  if env.config["cache"]
    write_sets(dict, joinpath(env.dir, "data", lib_name*".h5"))
  end
end

export contains
function contains(env::DLEnv, lib_name::String)
  if haskey(env._ext_event_libs, lib_name) && !isempty(env._ext_event_libs[lib_name])
    return true
  end
  return env.config["cache"] && isfile(_cachefile(env, lib_name))
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
    c_file = _cachefile(env, lib_name)
    if isfile(c_file)
      env._ext_event_libs[lib_name] = read_sets(c_file)
    else
      env._ext_event_libs[lib_name] = Dict()
    end
  end
end

function Base.delete!(env::DLEnv, lib_name::String)
  if haskey(env._ext_event_libs, lib_name)
    delete!(env._ext_event_libs, lib_name)
  end
  # Delete cache file (even if cache is set to false)
  file = _cachefile(env, lib_name)
  if isfile(file)
    rm(file)
    info(env, 2, "Deleted cached $lib_name")
  end
end

function _cachefile(env::DLEnv, lib_name::String)
  return joinpath(env.dir, "data", lib_name*".h5")
end


function network(env::DLEnv, name::String)
    dir = joinpath(env.dir, "models", name)
    isdir(dir) || mkdir(dir)
    return NetworkInfo(name, dir, get_properties(env, name))
end
