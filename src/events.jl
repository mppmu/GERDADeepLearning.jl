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

  function DLData(entries::Vector{EventLibrary})
    for e in entries
      @assert e != nothing
    end
    new(entries)
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


export waveforms
waveforms(lib::EventLibrary) = initialize(lib).waveforms
waveforms(data::DLData) = [waveforms(e) for e in data]

export sampling_rate
sampling_rate(lib::EventLibrary) = lib[:sampling_rate]
sampling_rate(data::DLData) = sampling_rate(data.entries[1])

export sampling_period
sampling_period(lib::EventLibrary) = 1/lib.sampling_rate
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


export filter_by_proxy
function filter_by_proxy(sets::Dict{Symbol,EventLibrary}, predicate_sets::Dict{Symbol,EventLibrary}, predicate_key, predicate)
  result = Dict{Symbol,EventLibrary}()
  for (key, lib) in sets
    indices = find(predicate, predicate_sets[key].labels[predicate_key])
    result[key] = lib[indices]
  end
  return result
end

export getindex
function getindex(events::EventLibrary, key::Symbol)
  if haskey(events.labels, key)
    return events.labels[key]
  else
    return events.prop[key]
  end
end

function getindex(events::EventLibrary, key::AbstractString)
  return getindex(events, Symbol(key))
end

function getindex(events::EventLibrary, selection)
  result = EventLibrary(events.waveforms[:,selection])
  for (key, value) in events.labels
    result.labels[key] = value[selection]
  end
  for (key,value) in events.prop
    result.prop[key] = value
  end
  return result
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
Base.string(data::DLData) = "DLData ($(length(data.entries)) libraries)"
Base.show(data::DLData) = println(string(data))

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




# function cat_events()
#
# end

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
  println("Equalizing $(name(events)) to $count counts. SSE: $(length(i_SSE)), MSE: $(length(i_MSE))")

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
