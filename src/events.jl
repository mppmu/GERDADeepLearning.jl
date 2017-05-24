# This file is a part of GERDADeepLearning.jl, licensed under the MIT License (MIT).

using HDF5
import Base: filter, length, getindex, haskey, copy, string, print, println, deepcopy
import HDF5: name
using Compat


@compat abstract type EventCollection
end

export EventLibrary
type EventLibrary <: EventCollection
  waveforms::Array{Float32, 2}
  labels::Dict{Symbol,Vector{Float32}} # same length as waveforms
  prop::Dict{Symbol,Any}

  EventLibrary(waveforms::Matrix{Float64}) = EventLibrary(convert(Matrix{Float32}, waveforms))
  EventLibrary(waveforms::Matrix{Float32}) = new(waveforms, 100e6, Dict(), Dict())
end

export DLData
type DLData <: EventCollection
  entries :: Vector{EventLibrary}

  function DLData(entries::Vector{EventLibrary})
    @assert length(entries) > 0
    new(entries)
  end
end

Base.start(data::DLData) = 1
Base.done(data::DLData, state) = length(data.entries) == state-1
Base.next(data::DLData, state) = data.entries[state], state+1

export waveforms
waveforms(events::EventLibrary) = events.waveforms
waveforms(data::DLData) = [waveforms(e) for e in data]

export sampling_rate
sampling_rate(lib::EventLibrary) = lib[:sampling_rate]
sampling_rate(data::DLData) = sampling_rate(data.entries[1])

export sampling_period
sampling_period(lib::EventLibrary) = 1/lib.sampling_rate
sampling_period(data::DLData) = sampling_period(data.entries[1])

export sample_size
sample_size(lib::EventLibrary) = size(lib.waveforms, 1)
sample_size(data::DLData) = sample_size(data.entries[1])


export filter
function filter(events::EventLibrary, predicate_key, predicate::Function)
  indices = find(predicate, events.labels[predicate_key])
  return events[indices]
end

# function filter(lib::EventLibrary, predicate_key::Symbol, )

function filter(sets::Dict{Symbol,EventLibrary}, predicate_key, predicate::Function)
  result = Dict{Symbol,EventLibrary}()
  for (key, lib) in sets
    result[key] = filter(lib, predicate_key, predicate)
  end
  return result
end

function filter(data::DLData, predicate_key, predicate::Function)
  if hasproperty(data, predicate_key)
    # Filter list of datasets
    filtered_indices = find(lib -> predicate(lib.props[predicate_key]), data.entries)
    return DLData(data.entries[filtered_indices])
  else
    # Filter individual datasets
    return DLData([filter(lib, predicate_key, predicate) for lib in data.entries])
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

export length
function length(events::EventLibrary)
  return size(events.waveforms, 2)
end

export totallength
function totallength(sets::Dict{Symbol,EventLibrary})
  result = 0
  for (key, events) in sets
    result += length(events)
  end
  return result
end

export haskey
function haskey(events::EventLibrary, key::Symbol)
  return haskey(events.labels, key) || haskey(events.prop, key)
end

export hasproperty
hasproperty(events::EventLibrary, key::Symbol) = haskey(events.prop, key)
hasproperty(data::DLData, key::Symbol) = hasproperty(data.entries[1])



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
  return "$(name(events)) ($(length(events)) events)"
end

function Base.summary(events::EventLibrary)
  return "$(name(events)) ($(length(events)) events)"
end

export name
function name(events::EventLibrary)
  return haskey(events, :name) ? events[:name] : "<unnamed>"
end

export setname!
function setname!(events::EventLibrary, name::String)
  events.prop[:name] = name
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
  labels = [_get_label(events[:E][i], peaks0, peaks1, half_window) for i in 1:length(events)]
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
  @assert length(data) == length(events)
  events.labels[label_name] = data
end
