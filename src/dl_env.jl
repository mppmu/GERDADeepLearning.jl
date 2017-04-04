# This file is a part of GERDADeepLearning.jl, licensed under the MIT License (MIT).

using JSON

import Base: get, contains, push!

export DLEnv
type DLEnv # immutable
  dir::AbstractString
  config::Dict
  _ext_event_libs::Dict{String,Dict{Symbol,EventLibrary}}

  DLEnv() = DLEnv("config")
  DLEnv(name::AbstractString) = DLEnv(abspath(""), name)
  function DLEnv(dir::AbstractString, name::AbstractString)
    f = open(joinpath(dir,"$name.json"), "r")
    dicttxt = readstring(f)  # file information to string
    dict = JSON.parse(dicttxt)  # parse and transform data
    env = new(dir, dict, Dict())
    setup(env)
    return env
  end
end

export get_properties
function get_properties(env::DLEnv, name::AbstractString)
  return get(env.config, name, Dict())
end

export set_properties
function set_properties(env::DLEnv, name::AbstractString, props::Dict)
  env.config[name] = props
end

export resolvepath
function resolvepath(env::DLEnv, path::AbstractString)
  if isabspath(path)
    return path
  else
    return joinpath(env.dir, path)
  end
end


export getdata
"""
Get the data, defined by the keylists in config.json.
The data is split into different data sets such as training and test.
This method returns a Dict{Symbol, EventLibrary} containing the different sets.
If the data has been loaded before, the same dictionary is returned.
If caching is active (set in config.json), the cached data will be read if available.
Otherwise a new cache file will be created.
If overwrite_existing is set to true, the original data is read and optionally cached.
"""
function getdata(env::DLEnv; targets::Array{String}=[])
  data = get(env, "data"; targets=targets) do
    # Else read original data
    println("Reading original data from $(env.config["path"])")
    keylist = FileKey[]
    setdict = _setdict(env)
    set_names = collect(keys(setdict))
    set_sizes = []
    for (i,keylist_path) in enumerate(env.config["keylists"])
      new_keys = parse_keylist(resolvepath(env, keylist_path))
      push!(keylist, new_keys...)
      push!(set_sizes, fill([setdict[setname][i] for setname in set_names], length(new_keys))...)
    end
    sets = read_tier_W(env.config["path"], keylist,
        set_names=set_names, set_sizes=set_sizes,
        select_channels=env.config["detectors"])
    return sets
  end
  if data != nothing
    for (n, set) in data println(set) end
  end
  return data
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
function get(compute, env::DLEnv, lib_name::String; targets::Array{String}=[])
  if !isempty(targets) && containsall(env, targets)
    println("Skipping retrieval of '$lib_name'.")
    return nothing
  end
  return get(compute, env, lib_name)
end

function get(compute, env::DLEnv, lib_name::String)
  if contains(env, lib_name)
    println("Retrieving '$lib_name' from cache.")
    return get(env, lib_name)
  else
    println("Computing '$lib_name'...")
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

function _cachefile(env::DLEnv, lib_name::String)
  return joinpath(env.dir, "data", lib_name*".h5")
end


function network(env::DLEnv, name::String)
    return NetworkInfo(name, joinpath(env.dir, "models"), get_properties(env, name))
end
