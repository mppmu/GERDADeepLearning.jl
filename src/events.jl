# This file is a part of GERDADeepLearning.jl, licensed under the MIT License (MIT).

using HDF5
import Base: filter, length, getindex, haskey, copy, string, print, println, deepcopy
import HDF5: name
using Compat


@compat abstract type EventCollection
end

export EventLibrary
type EventLibrary <: EventCollection
  # EventLibraries can be lazily initialized.
  # Possible initializers: load from file, filter by label
  initialization_function::Union{Function,Void}

  waveforms::Array{Float32, 2}
  labels::Dict{Symbol,Vector{Float32}} # same length as waveforms
  prop::Dict{Symbol,Any}

  EventLibrary(init::Function) = new(init, zeros(Float32, 0,0), Dict{Symbol,Vector{Float32}}(), Dict{Symbol,Any}())

  EventLibrary(waveforms::Matrix{Float64}) = EventLibrary(convert(Matrix{Float32}, waveforms))

  EventLibrary(waveforms::Matrix{Float32}) = new(nothing, waveforms, Dict{Symbol,Vector{Float32}}(), Dict{Symbol,Any}())
end

export DLData
type DLData <: EventCollection
  entries :: Vector{EventLibrary}
  dir :: Union{AbstractString,Void}

  function DLData(entries::Vector{EventLibrary})
    for e in entries
      @assert e != nothing
    end
    new(entries, nothing)
  end
end


function _set_shallow(lib::EventLibrary, from::EventLibrary)
  lib.initialization_function = from.initialization_function
  lib.waveforms = from.waveforms
  lib.labels = from.labels
  lib.prop = from.prop
end

is_initialized(lib::EventLibrary) = lib.initialization_function == nothing

export initialize
function initialize(lib::EventLibrary)
  if lib.initialization_function != nothing
    lib.initialization_function(lib)
    lib.initialization_function = nothing
  end
  return lib
end

function dispose(lib::EventLibrary)
  lib.waveforms = zeros(Float32, 0, 0)
  empty!(lib.labels)
end
dispose(data::DLData) = for lib in data didpose(lib) end

function initialize(data::DLData)
  for lib in data
    initialize(lib)
  end
  return data
end

Base.start(data::DLData) = 1
Base.done(data::DLData, state) = length(data.entries) == state-1
Base.next(data::DLData, state) = data.entries[state], state+1
Base.length(data::DLData) = length(data.entries)

export flatten
function flatten(data::DLData)
  if length(data.entries) == 0
    return nothing
  end
  if length(data.entries) == 1
    return data.entries[1]
  end
  return cat_events(data.entries...)
end

export waveforms
waveforms(lib::EventLibrary) = initialize(lib).waveforms
waveforms(data::DLData) = waveforms(flatten(data))

export sampling_rate
sampling_rate(lib::EventLibrary) = lib[:sampling_rate]
sampling_rate(data::DLData) = sampling_rate(data.entries[1])

export sampling_period
sampling_period(lib::EventLibrary) = 1/sampling_rate(lib)
sampling_period(data::DLData) = sampling_period(data.entries[1])

export sample_size
sample_size(lib::EventLibrary) = size(initialize(lib).waveforms, 1)
sample_size(data::DLData) = sample_size(data.entries[1])


export filter

filter(lib::EventCollection, key::Symbol, value::Union{AbstractString,Number}) = filter(lib, key, x -> x==value)
filter{T<:Union{AbstractString,Number}}(lib::EventCollection, key::Symbol, values::Vector{T}) = filter(lib, key, x -> x in values)

function filter(lib::EventLibrary, predicate_key::Symbol, predicate::Function)
  if is_initialized(lib)
    indices = find(predicate, lib.labels[predicate_key])
    return lib[indices]
  else
    result = copy(lib) # shallow copy
    result.initialization_function = lib2 -> _set_shallow(lib2, filter(initialize(lib), predicate_key, predicate))
    return result
  end
end

function filter(data::DLData, predicate_key::Symbol, predicate::Function)
  if hasproperty(data, predicate_key)
    # Filter list of datasets
    filtered_indices = find(lib -> predicate(lib.prop[predicate_key]), data.entries)
    return DLData(data.entries[filtered_indices])
  else
    # Filter individual datasets
    return DLData([filter(lib, predicate_key, predicate) for lib in data])
  end
end


Base.filter!(lib::EventCollection, key::Symbol, value::Union{AbstractString,Number}) = filter!(lib, key, x -> x==value)
Base.filter!{T<:Union{AbstractString,Number}}(lib::EventCollection, key::Symbol, values::Vector{T}) = filter!(lib, key, x -> x in values)

function Base.filter!(lib::EventLibrary, predicate_key::Symbol, predicate::Function)
  if is_initialized(lib)
    indices = find(predicate, lib.labels[predicate_key])
    lib.waveforms = lib.waveforms[:,indices]
    for label in keys(lib.labels)
      lib.labels[label] = lib.labels[label][indices]
    end
    return lib
  else
    prev_initialization = lib.initialization_function
    lib.initialization_function = lib2 -> begin
      lib2.initialization_function = nothing
      prev_initialization(lib2)
      filter!(lib2, predicate_key, predicate)
    end
    return lib
  end
end

function Base.filter!(data::DLData, predicate_key::Symbol, predicate::Function)
  if hasproperty(data, predicate_key)
    # Filter list of datasets
    filtered_indices = find(lib -> predicate(lib.prop[predicate_key]), data.entries)
    data.entries = data.entries[filtered_indices]
    return data
  else
    # Filter individual datasets
    for lib in data
      filter!(lib, predicate_key, predicate)
    end
  end
end


export filter_by_proxy
function filter_by_proxy(lib::EventLibrary, predicate_lib::EventLibrary, predicate_key, predicate)
  @assert eventcount(lib) == eventcount(predicate_lib)
  indices = find(predicate, predicate_lib[predicate_key])
  return lib[indices]
end
function filter_by_proxy(data::DLData, predicate_data::DLData, predicate_key, predicate)
  return DLData([filter_by_proxy(data.entries[i], predicate_data.entries[i], predicate_key, predicate) for i in 1:length(data.entries)])
end

export getindex
function getindex(lib::EventLibrary, key::Symbol)
  if haskey(lib.prop, key)
    return lib.prop[key]
  else
    return initialize(lib).labels[key]
  end
end
getindex(data::DLData, key::Symbol) = flatten(data)[key]

function getindex(events::EventLibrary, key::AbstractString)
  return getindex(events, Symbol(key))
end

function getindex(events::EventLibrary, selection)
  initialize(events)
  result = EventLibrary(events.waveforms[:,selection])
  for (key, value) in events.labels
    result.labels[key] = value[selection]
  end
  for (key,value) in events.prop
    result.prop[key] = value
  end
  return result
end

function getindex{T<:Union{AbstractString,Number}}(events::EventCollection, predicates::Pair{Symbol,T}...)
  for predicate in predicates
    events = filter(events, predicate[1], predicate[2])
  end
  return events
end

export eventcount
eventcount(events::EventLibrary) = size(initialize(events).waveforms, 2)
eventcount(data::DLData) = sum([eventcount(lib) for lib in data])

Base.haskey(events::EventLibrary, key::Symbol) = haskey(events.labels, key) || haskey(events.prop, key)
Base.haskey(data::DLData, key::Symbol) = length(data) > 0 && haskey(data.entries[1])

export hasproperty
hasproperty(events::EventLibrary, key::Symbol) = haskey(events.prop, key)
hasproperty(data::DLData, key::Symbol) = length(data) > 0 && hasproperty(data.entries[1], key)



export copy
function copy(events::EventLibrary)
  result = EventLibrary(events.waveforms)
  for (key, value) in events.labels
    result.labels[key] = value
  end
  for (key,value) in events.prop
    result.prop[key] = value
  end
  return result
end

function copy(data::DLData)
  return DLData([copy(lib) for lib in data])
end

export deepcopy
function deepcopy(events::EventLibrary)
  result = EventLibrary(copy(events.waveforms))
  for (key, value) in events.labels
    result.labels[key] = copy(value)
  end
  for (key,value) in events.prop
    result.prop[key] = copy(value)
  end
  return result
end


function Base.string(events::EventLibrary)
  if is_initialized(events)
    return "$(name(events)) ($(eventcount(events)) events)"
  else
    return "$(name(events)) (not yet initialized)"
  end
end
Base.string(data::DLData) = "DLData ($(length(data.entries)) subsets)"
Base.show(data::DLData) = println(data)
Base.show(io::IO, data::DLData) = println(io, string(data))
Base.print(io::IOBuffer, data::DLData) = println(io, string(data))
Base.print(data::DLData) = print(string(data))
function Base.println(data::DLData)
  println(string(data))
  for lib in data
    println(string(lib))
  end
end

Base.summary(events::EventCollection) = string(events)

export name
name(events::EventLibrary) = haskey(events, :name) ? events[:name] : "<unnamed>"
name(data::DLData) = [name(lib) for lib in data]

export setname!
function setname!(events::EventLibrary, name::String)
  events.prop[:name] = name
end

function setname!(data::DLData, names::Vector{String})
  for i in 1:length(data.entries)
    setname!(data.entries[i], names[i])
  end
end

# export convert
# function convert(target::Type{String}, events::EventLibrary)
#   if target == String
#     return string(events)
#   end
#   throw(MethodError(convert, target, events))
# end

export get_classifiers
function get_classifiers(events::EventLibrary)
  if haskey(events.prop, :classifier_names)
    return events.prop[:classifier_names]
  else
    return String[]
  end
end

export push_classifier!
function push_classifier!(events::EventLibrary, clname::AbstractString)
  if haskey(events.prop, :classifier_names)
    push!(events.prop[:classifier_names], clname)
  else
    events.prop[:classifier_names] = [clname]
  end
end



export cat_events
function cat_events(libs::EventLibrary...)
  @assert length(libs) > 0
  nonempty = find(lib->eventcount(lib)>0, libs)

  result = EventLibrary(identity)
  result.initialization_function = nothing

  # Cat waveforms
  if length(nonempty) > 0
    result.waveforms = hcat([lib.waveforms for lib in libs[nonempty]]...)
    # Cat labels
    for key in keys(libs[1].labels)
      result.labels[key] = vcat([lib.labels[key] for lib in libs[nonempty]]...)
    end
  else
    result.waveforms = libs[1].waveforms
    result.labels = copy(libs[1]).labels
  end

  # Adopt shared property values
  for key in keys(libs[1].prop)
    values = [lib.prop[key] for lib in libs]
    if !any(x->x!=values[1], values)
      result.prop[key] = values[1]
    end
  end
  return result
end


function Base.split(data::DLData, datasets::Dict{AbstractString,Vector{AbstractFloat}})
    result = DLData(EventLibrary[])
    for lib in copy(data.entries)
      split_result = split(lib, datasets)
      for (set_name, lib) in split_result
        push!(result.entries, lib)
      end
    end
    return result
end


# function split_uninitialized(lib::EventLibrary, fractions::Dict{AbstractString,Vector{AbstractFloat}})
# end

function Base.split(lib::EventLibrary, fractions::Dict{AbstractString,Vector{AbstractFloat}})
  keylist_ids = lib[:keylist] # 1-based
  if eventcount(lib) > 0
    kl_count = Int(maximum(keylist_ids))
  else
    kl_count = 0
  end

  dsets = Dict{AbstractString,Vector{EventLibrary}}()
  for (dset_name, dset_sizes) in fractions
    dsets[dset_name] = EventLibrary[]
  end

  # for each keylist split separately
  for keylist_id in 1:kl_count
    keylist_indices = find(x->x==keylist_id, keylist_ids)
    N = length(keylist_indices)
    index_perm = randperm(N)

    depleted_fraction = 0.0
    # split shuffled indices into data sets
    for (dset_name, dset_sizes) in fractions
      dset_frac = dset_sizes[keylist_id]
      start_i = Int(round(depleted_fraction*N)) + 1
      end_i = Int(round((depleted_fraction+dset_frac)*N))
      dset_indices = keylist_indices[index_perm[start_i : end_i]]
      push!(dsets[dset_name], lib[dset_indices])
      depleted_fraction += dset_frac
    end
    if !(depleted_fraction ≈ 1)
      info("Some events were not assigned to any data set during split.")
    end
  end

  # Combine different keylists
  result = Dict{AbstractString,EventLibrary}()
  for dset_name in keys(fractions)
    if eventcount(lib) > 0
      result[dset_name] = cat_events(dsets[dset_name]...)
    else
      result[dset_name] = copy(lib)
    end
    result[dset_name].prop[:set] = dset_name
    setname!(result[dset_name], name(result[dset_name])*"_"*dset_name)
  end
  return result
end

export label_energy_peaks
function label_energy_peaks(events::EventLibrary, label_key=:SSE, peaks0=[1620.7], peaks1=[1592.5], half_window=2.0)
  labels = [_get_label(events[:E][i], peaks0, peaks1, half_window) for i in 1:eventcount(events)]
  events.labels[label_key] = labels
  return events
end

function _get_label(energy, peaks0, peaks1, half_window)
  for peak0 in peaks0
    if abs(energy-peak0) <= half_window
      return 0
    end
  end
  for peak1 in peaks1
    if(abs(energy-peak1)) <= half_window
      return 1
    end
  end
  return -1
end

export equalize_counts_by_label
function equalize_counts_by_label(events::EventLibrary, label_key=:SSE)
  if !haskey(events, label_key)
    label_energy_peaks(events, label_key)
  end
  labels = events.labels[label_key]
  i_SSE = find(x -> x==1, labels)
  i_MSE = find(x -> x==0, labels)
  count = min(length(i_SSE), length(i_MSE))
  info("Equalizing $(name(events)) to $count counts. SSE: $(length(i_SSE)), MSE: $(length(i_MSE))")

  # trim and shuffle
  used_indices = [i_SSE[1:count];i_MSE[1:count]]
  shuffled_indices = used_indices[randperm(length(used_indices))]

  return events[shuffled_indices], shuffled_indices
end


export put_label!
function put_label!(events::EventLibrary, label_name::Symbol, data::Vector{Float32})
  @assert length(data) == eventcount(events)
  events.labels[label_name] = data
end
