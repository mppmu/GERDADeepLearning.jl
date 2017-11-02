# This file is a part of GERDADeepLearning.jl, licensed under the MIT License (MIT).

using HDF5
import Base: filter, length, getindex, haskey, copy, string, print, deepcopy
using Compat


export EventCollection
@compat abstract type EventCollection
end

type NoSuchEventException <: Exception
end

export EventLibrary
type EventLibrary <: EventCollection
  # EventLibraries can be lazily initialized.
  # Possible initializers: load from file, filter by label
  initialization_function::Union{Function,Void}

  waveforms::Array{Float32, 2}
  labels::Dict{Symbol,Vector} # same length as waveforms
  prop::Dict{Symbol,Any}

  EventLibrary(init::Function) = new(init, zeros(Float32, 0,0), Dict{Symbol,Vector}(), Dict{Symbol,Any}())

  EventLibrary(waveforms::Matrix{Float64}) = EventLibrary(convert(Matrix{Float32}, waveforms))

  EventLibrary(waveforms::Matrix{Float32}) = new(nothing, waveforms, Dict{Symbol,Vector}(), Dict{Symbol,Any}())
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

export no_events
no_events() = DLData(EventLibrary[])

function Base.cat(libs::Vector{EventLibrary})
    if length(libs) == 1
        return libs[1]
    end
    return DLData(libs)
end
Base.cat(libs::EventLibrary...) = cat([lib for lib in libs])



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

export dispose
function dispose(lib::EventLibrary)
  lib.waveforms = zeros(Float32, 0, 0)
  empty!(lib.labels)
end
dispose(data::DLData) = for lib in data dispose(lib) end

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

Base.start(lib::EventLibrary) = false
Base.done(lib::EventLibrary, state) = state
Base.next(lib::EventLibrary, state) = lib, true
Base.length(lib::EventLibrary) = 1


export flatten
function flatten(data::DLData)
  if length(data.entries) == 0
    return data
  end
  if length(data.entries) == 1
    return data.entries[1]
  end
  return cat_events(data.entries...)
end
flatten(lib::EventLibrary) = lib

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

export sample_times
sample_times(events::EventCollection) = linspace(0, (sample_size(events)-1)*sampling_period(events), sample_size(events))

export filter

filter(lib::EventCollection, key::Symbol, value::Union{AbstractString,Number}) = filter(lib, key, x -> x==value)
filter{T<:Union{AbstractString,Number}}(lib::EventCollection, key::Symbol, values::Vector{T}) = filter(lib, key, x -> x in values)

function filter(lib::EventLibrary, predicate_key::Symbol, predicate::Function)
    if haskey(lib.prop, predicate_key)
        propval = lib.prop[predicate_key]
        if !predicate(propval)
            return nothing
        else
            return lib
        end
    end
    
  if is_initialized(lib)
    indices = find(predicate, lib.labels[predicate_key])
    return lib[indices]
  else
    result = copy(lib) # shallow copy
    result.initialization_function = lib2 -> _set_shallow(lib2, filter(initialize(lib), predicate_key, predicate))
    delete!(result.prop, :eventcount)
    return result
  end
end

function filter(data::DLData, predicate_key::Symbol, predicate::Function)
  if hasproperty(data, predicate_key)
    # Filter list of datasets
    filtered_indices = find(lib -> predicate(lib.prop[predicate_key]), data.entries)
    if length(filtered_indices) == 1
        return data.entries[filtered_indices[1]]
    end
    return DLData(data.entries[filtered_indices])
  else
    # Filter individual datasets
    return DLData([filter(lib, predicate_key, predicate) for lib in data])
  end
end


Base.filter!(lib::EventCollection, key::Symbol, value::Union{AbstractString,Number}) = filter!(lib, key, x -> x==value)
Base.filter!{T<:Union{AbstractString,Number}}(lib::EventCollection, key::Symbol, values::Vector{T}) = filter!(lib, key, x -> x in values)

function Base.filter!(lib::EventLibrary, predicate_key::Symbol, predicate::Function)
    # First check whether it's a property
  if haskey(lib.prop, predicate_key)
        propval = lib.prop[predicate_key]
        if !predicate(propval)
            lib.waveforms = zeros(Float32, 0, 0)
            lib.initialization_function = nothing
            for label in keys(lib.labels)
              lib.labels[label] = zeros(Float32, 0)
            end
        end
        return lib
    end

    # Else it has to be a label
  if is_initialized(lib)
    indices = find(predicate, lib.labels[predicate_key])
    lib.waveforms = lib.waveforms[:,indices]
    for label in keys(lib.labels)
      lib.labels[label] = lib.labels[label][indices]
    end
  else
    prev_initialization = lib.initialization_function
    lib.initialization_function = lib2 -> begin
      lib2.initialization_function = nothing
      prev_initialization(lib2)
      filter!(lib2, predicate_key, predicate)
    end
  end
  delete!(lib.prop, :eventcount)
  return lib
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


function filter(lib::EventLibrary, predicate_keys::Vector{Symbol}, predicate::Function)
    if is_initialized(lib)
        indices = Int64[]
        arg_arrays = [lib.labels[predicate_key] for predicate_key in predicate_keys]
        for i in 1:eventcount(lib)
            args = [arg_array[i] for arg_array in arg_arrays]
            if predicate(args...)
                push!(indices, i)
            end
        end
        return lib[indices]
    else
        result = copy(lib) # shallow copy
        result.initialization_function = lib2 -> _set_shallow(lib2, filter(initialize(lib), predicate_keys, predicate))
        delete!(result.prop, :eventcount)
        return result
    end
end

function filter(data::DLData, predicate_keys::Vector{Symbol}, predicate::Function)
    return cat([filter(lib, predicate_keys, predicate) for lib in data])
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
    if key == :wf || key == :waveforms || key == :waveform
        return waveforms(lib)
    end
  if haskey(lib.prop, key)
    return lib.prop[key]
  else
    return initialize(lib).labels[key]
  end
end
function getindex(data::DLData, key::Symbol)
    if length(data) == 0
        return []
    end
    if haskey(data.entries[1].prop, key)
        return [lib[key] for lib in data]
    end
    return flatten(data)[key]
end

function getindex(events::EventLibrary, key::AbstractString)
  return getindex(events, Symbol(key))
end

function getindex(events::EventLibrary, selection::Union{Range, Integer, Array{Int64}, Array{Int32}})
  initialize(events)
  result = EventLibrary(events.waveforms[:,selection])
  for (key, value) in events.labels
    result.labels[key] = value[selection]
  end
  for (key,value) in events.prop
    result.prop[key] = value
  end
  delete!(result.prop, :eventcount)
  return result
end
getindex(data::DLData, selection::Union{UnitRange,Integer}) = flatten(data)[selection]

function getindex{T<:Union{AbstractString,Number}}(events::EventCollection, predicates::Pair{Symbol,T}...)
  for predicate in predicates
    events = filter(events, predicate[1], predicate[2])
  end
  return events
end

export eventcount
function eventcount(events::EventLibrary)
    if size(events.waveforms, 2) > 0
        return size(events.waveforms, 2)
    elseif haskey(events, :eventcount)
        return events[:eventcount]
    else
        return size(initialize(events).waveforms, 2)
    end
end
function eventcount(data::DLData)
    if length(data.entries) == 0
        return 0
    end
    return sum([eventcount(lib) for lib in data.entries])
end

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
  result.initialization_function = events.initialization_function
  for (key, value) in events.labels
    result.labels[key] = copy(value)
  end
  for (key,value) in events.prop
    result.prop[key] = deepcopy(value)
  end
  return result
end


function Base.string(events::EventLibrary)
  if is_initialized(events)
    return "$(events[:name]) ($(eventcount(events)) events)"
  else
    return "$(events[:name]) (not yet initialized)"
  end
end
Base.println(lib::EventLibrary) = println(string(lib))
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

export eventinfo
function eventinfo(events::EventCollection, index::Integer)
    lib, i = unravel_index(events, index)
    println("Event properties:")
    for (key, arr) in lib.labels
        println("  $key: $(arr[i])")
    end
    println("General properties:")
    for (key, val) in lib.prop
        println("  $key: $val")
    end
end

export setname!
function setname!(events::EventLibrary, name::String)
  events.prop[:name] = name
end

function setname!(data::DLData, names::Vector{String})
  for i in 1:length(data.entries)
    setname!(data.entries[i], names[i])
  end
end

function unravel_index(events::EventCollection, index::Integer)
    total = 0
    for lib in events
        if index > total+eventcount(lib)
            total += eventcount(lib)
        else
            return lib, index-total
        end
    end
    return throw(NoSuchEventException())
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
    result.waveforms = hcat([waveforms(lib) for lib in libs[nonempty]]...) # this initializes the entries
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
    delete!(result.prop, :eventcount)
  return result
end


function Base.sort(lib::EventLibrary, labelkey::Symbol)
    p = sortperm(lib[labelkey])
    return lib[p]
end
Base.sort(data::DLData, labelkey::Symbol) = sort(flatten(data), labelkey)



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
    index_perm = randperm(MersenneTwister(0), N)

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
  info("Equalizing $(events[:name]) to $count counts. SSE: $(length(i_SSE)), MSE: $(length(i_MSE))")

  # trim and shuffle
  used_indices = [i_SSE[1:count];i_MSE[1:count]]
  shuffled_indices = used_indices[randperm(length(used_indices))]

  return events[shuffled_indices], shuffled_indices
end


export put_label!
function put_label!(events::EventLibrary, label_name::Symbol, data::Vector)
  @assert length(data) == eventcount(events)
  events.labels[label_name] = data
end


function Base.keys(lib::EventLibrary)
    return vcat(collect(keys(lib.labels)), collect(keys(lib.prop)))
end
function Base.keys(data::DLData)
    if length(data.entries) == 0
        return Symbol[]
    else
        return keys(data.entries[1])
    end
end


export detectors
function detectors(data::DLData)
    return unique([lib[:detector_name] for lib in data])
end
detectors(lib::EventLibrary) = [lib[:detector_name]]




export SSE_at
function SSE_at(lib::EventLibrary, energy::Real)
    if startswith(lib[:detector_name], "GD")
        SSE_indices = find(AoE->(AoE>0.98)&&(AoE<1.02), lib[:AoE])
    else
        SSE_indices = find(c->c==0, lib[:ANN_mse_class])
    end
    sorted_indices = sortperm(abs.(lib[:E].-energy))
    for idx in sorted_indices
        if idx in SSE_indices
            return idx
        end
    end
    throw(NoSuchEventException())
end
SSE_at(data::DLData, energy::Real) = SSE_at(flatten(data), energy)

export MSE_at
function MSE_at(lib::EventLibrary, energy::Real; AoE_cut=0.7)
    if startswith(lib[:detector_name], "GD")
        MSE_indices = find(AoE->(AoE<AoE_cut)&&(AoE!=0), lib[:AoE])
    else
        MSE_indices = find(c->c==1, lib[:ANN_mse_class])
    end
    sorted_indices = sortperm(abs.(lib[:E].-energy))
    for idx in sorted_indices
        if idx in MSE_indices
            return idx
        end
    end
    throw(NoSuchEventException())
end
MSE_at(data::DLData, energy::Real) = MSE_at(flatten(data), energy)


export any_at
function any_at(lib::EventCollection, energy::Real)
    return findmin(abs.(lib[:E].-energy))[2]
end


export equal_event_count_edges
function equal_event_count_edges(events::EventCollection, label::Symbol; events_per_bin::Real=2000)
    labels = sort(events[label])
    total_event_count = eventcount(events)
    equal_count_edges = []
    i = 0
    while (i*events_per_bin) < total_event_count
        push!(equal_count_edges, labels[i*events_per_bin+1])
        i += 1
    end
    return equal_count_edges
end

export lookup_property
function lookup_property(source::EventLibrary, sourceindex::Integer, target::EventCollection, propkey::Symbol)
    time = source[:timestamp][sourceindex]
    energy = source[:E][sourceindex]
    for lib in target
        t_times = lib[:timestamp]
        t_energies = lib[:E]
        for i in 1:eventcount(lib)
            if (t_times[i] == time) && (t_energies[i] == energy)
                return lib[i:i][propkey]
            end
        end
    end
    return nothing
end

