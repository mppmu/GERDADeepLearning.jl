# This file is a part of GERDADeepLearning.jl, licensed under the MIT License (MIT).

using HDF5
import Base: filter, length, getindex, haskey, copy, string, print, println, deepcopy
import HDF5: name


default_labels = [:aoeValues, :aoeClasses, :E]


export EventLibrary
type EventLibrary
  waveforms::Array{Float32, 2}
  sampling_rate::Number # in Hz
  labels::Dict{Symbol,Vector{Float32}} # same length as waveforms
  prop::Dict{Symbol,Any}

  EventLibrary(waveforms::Matrix{Float64}) = EventLibrary(convert(Matrix{Float32}, waveforms))
  EventLibrary(waveforms::Matrix{Float32}) = new(waveforms, 100e6, Dict(), Dict())
end


export waveforms
function waveforms(events::EventLibrary)
  return events.waveforms
end

export sampling_rate
function sampling_rate(lib::EventLibrary)
  return lib.sampling_rate
end

export sampling_period
function sampling_period(lib::EventLibrary)
  return 1/lib.sampling_rate
end

export samples
function samples(lib::EventLibrary)
  return size(lib.waveforms, 1)
end

export read_sets
function read_sets(filepath)
  result::Dict{Symbol, EventLibrary} = Dict()

  ifile = h5open(filepath, "r")
  for setentry in ifile
    setname = Symbol(name(setentry)[2:end])
    setdata = read(setentry)

    waveforms = setdata["waveforms"]
    events = EventLibrary(waveforms)
    events.labels = _str_to_sym_dict(setdata["labels"])
    if haskey(setdata, "prop")
      events.prop = _str_to_sym_dict(setdata["prop"])
    else
      events.prop = Dict()
    end
    result[setname] = events
  end

  return result
end


function _str_to_sym_dict(dict)
  result = Dict{Symbol, Any}()
  for (key,value) in dict
    result[Symbol(key)] = value
  end
  return result
end

export read_events
function read_events(filepath)
  ifile = h5open(filepath, "r")

  waveforms = read(ifile, "waveforms")

  events = EventLibrary(waveforms)

  for entry in ifile
    data = read(entry)
    key = name(entry)[2:end]
    if startswith(key,"label_")
      events.labels[Symbol(key[7:end])] = data
    end
    if startswith(key,"prop_")
      events.prop[Symbol(key[6:end])] = data
    end
  end

  close(ifile)
  return events
end


export write_sets
function write_sets(sets::Dict{Symbol,EventLibrary}, filepath::AbstractString)
  h5open(filepath, "w") do file
    for (setname, events) in sets
      write(file, "$setname/waveforms", events.waveforms)
      for (key,value) in events.labels
        write(file, "$setname/labels/"*string(key), value)
      end
      for(key,value) in events.prop
        try
          write(file, "$setname/prop/"*string(key), value)
        catch err
          info("Cannot write property $key: $err")
        end
      end
    end
  end
end

export write_events
function write_events(events::EventLibrary, filepath::AbstractString)
  h5open(filepath, "w") do file
    write(file, "waveforms", events.waveforms)
    for (key,value) in events.labels
      write(file, "label_"*string(key), value)
    end
    for(key,value) in events.prop
      write(file, "prop_"*string(key), value)
    end
  end
end


export filter
function filter(events::EventLibrary, predicate_key=:E, predicate=e->((e>600) && (e<10000)))
  list = events.labels[predicate_key]
  indices = find(predicate, list)
  return events[indices]
end

function filter(sets::Dict{Symbol,EventLibrary}, predicate_key, predicate)
  result = Dict{Symbol,EventLibrary}()
  for (key, lib) in sets
    result[key] = filter(lib, predicate_key, predicate)
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

export haskey
function haskey(events::EventLibrary, key::Symbol)
  return haskey(events.labels, key) || haskey(events.prop, key)
end


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
  events.labels[label_name] = data
end
