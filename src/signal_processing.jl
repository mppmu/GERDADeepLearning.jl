# This file is a part of GERDADeepLearning.jl, licensed under the MIT License (MIT).

using DSP, MultiThreadingTools


export preprocess_transform
function preprocess_transform(env::DLEnv, lib::EventLibrary, steps::Vector{String}; copyf=deepcopy)
  initialize(lib)
  result = copyf(lib)
  setname!(result, name(lib)*"_preprocessed")
  put_label!(result, :FailedPreprocessing, zeros(Float32, eventcount(lib)))
  result.prop[:preprocessing] = steps

  # perform the steps
  for (i,step) in enumerate(steps)
    info(env, 3, "Preprocesing $step on $(eventcount(result)) events...")
    pfunction = eval(parse(step))
    result = pfunction(result)
  end

  failed_count = length(find(f -> f != 0, result[:FailedPreprocessing]))
  if failed_count > 0
    info(env, 1, "Preprocesing failed for $failed_count events in $(name(lib)). These have been tagged with the label :FailedPreprocessing = 1")
  end

  return result
end


export charge_pulses
function charge_pulses(events::EventLibrary; create_new=true)
  if events[:waveform_type] == "charge"
    return events
  elseif events[:waveform_type] == "current"
    if create_new
      events = deepcopy(events)
    end
    return integrate(events)
  elseif events[:waveform_type] == "raw"
    if create_new
      events = deepcopy(events)
    end
    return raw_to_charge(events)
  else
    throw(ArgumentError("EventLibrary must be of type current or charge."))
  end
end

export current_pulses
function current_pulses(events::EventLibrary; create_new=true)
  if events[:waveform_type] == "current"
    return events
  elseif events[:waveform_type] == "charge"
    if create_new
      events = deepcopy(events)
    end
    return differentiate(events)
  elseif events[:waveform_type] == "raw"
    if create_new
      events = deepcopy(events)
    end
    return current_pulses(raw_to_charge(events); create_new=false)
  else
    throw(ArgumentError("EventLibrary must be of type current or charge."))
  end
end

function current_pulses(libs::Dict{Symbol,EventLibrary}; create_new=true)
  return mapvalues(libs, current_pulses, create_new)
end

function raw_to_charge(events::EventLibrary)
  if events[:waveform_type] != "raw"
    throw(ArgumentError("EventLibrary must contain pulses of type raw"))
  end
  events.waveforms = - events.waveforms
  events.prop[:waveform_type] = "charge"
  return events
end


export baseline
function baseline(events::EventLibrary)
  events = charge_pulses(events; create_new=false)

  bl_size = Int64(round(sample_size(events) / 5))
  weights = hamming(bl_size)
  weights /= sum(weights)

  bl_levels = zeros(eventcount(events))
  bl_std = zeros(eventcount(events))

  @everythread for i in threadpartition(1:eventcount(events))
    bl_level = dot(events.waveforms[1:bl_size, i], weights)
    bl_levels[i] = bl_level
    events.waveforms[:, i] -= bl_level
    bl_std[i] = std(events.waveforms[1:bl_size, i])
  end

  put_label!(events, :baseline_level, bl_levels)
  put_label!(events, :baseline_std, bl_std)
  return events
end

export HE
function HE(events::EventLibrary; cut=1000)
  return filter(events, :E, E -> E>cut)
end

export align_peaks
function align_peaks(events::EventLibrary; target_length=256)
  currents = current_pulses(events; create_new=true)

  s = sample_size(events)
  half = Int64(target_length/2)
  rwf = zeros(Float32, target_length, eventcount(events))

  @everythread for i in threadpartition(1:eventcount(events))
    max_index = findmax(currents.waveforms[:,i])[2]
    if (max_index < half) || (max_index > s - half)
      events[:FailedPreprocessing][i] = 1
    else
      rwf[:,i] = events.waveforms[(max_index-half+1) : (max_index+half) , i]
    end
  end

  events.waveforms = rwf

  return events
end

export align_midpoints
function align_midpoints(events::EventLibrary; center_y=0.5, target_length=256)
  charges = charge_pulses(events; create_new=true)

  s = sample_size(events)
  half = Int64(target_length/2)
  rwf = zeros(Float32, target_length, eventcount(events))

  @everythread for i in threadpartition(1:eventcount(events))
    index = findmin(abs.(charges.waveforms[:,i] - center_y))[2]
    if (index < half) || (index > s - half)
      events[:FailedPreprocessing][i] = 1
    else
      rwf[:,i] = events.waveforms[(index-half+1) : (index+half) , i]
    end
  end

  events.waveforms = rwf

  return events
end

export extract_noise
function extract_noise(events::EventLibrary; target_length=256)
  s = sample_size(events)
  rwf = zeros(Float32, target_length, eventcount(events))

  @everythread for i in threadpartition(1:eventcount(events))
    rwf[:,i] = events.waveforms[(s-target_length+1):s , i]
    rwf[:,i] -= mean(rwf[:,i])
  end

  events.waveforms = rwf

  return events
end

export normalize_energy
function normalize_energy(events::EventLibrary; value=1)
  charges = charge_pulses(events; create_new=true)

  top_size = Int64(round(sample_size(events) / 5))
  weights = hamming(top_size)
  weights /= sum(weights)

  top_levels = zeros(Float32, eventcount(events))

  @everythread for i in threadpartition(1:eventcount(events))
    top_level = dot(charges.waveforms[(end-top_size+1) : end, i], weights)
    top_levels[i] = top_level
    events.waveforms[:,i] *= value / top_level
  end

  put_label!(events, :top_level, top_levels)
  return events
end

export integrate
function integrate(events::EventLibrary)
    events = initialize(events)
  @everythread for i in threadpartition(1:eventcount(events))
    events.waveforms[:,i] = integrate_array(events.waveforms[:,i])
  end
  if events[:waveform_type] == "charge"
    events.prop[:waveform_type] = "other"
  elseif events[:waveform_type] == "current"
    events.prop[:waveform_type] = "charge"
  end
  return events
end


function integrate_array(a)
  b = copy(a)
  for i in 2:length(a)
    b[i] += b[i-1]
  end
  return b
end


export differentiate
function differentiate(events::EventLibrary)
    events = initialize(events)
  @everythread for i in threadpartition(1:eventcount(events))
      events.waveforms[:,i] = gradient(events.waveforms[:,i])
  end
  if events[:waveform_type] == "current"
    events.prop[:waveform_type] = "other"
  elseif events[:waveform_type] == "charge"
    events.prop[:waveform_type] = "current"
  end
  return events
end


export scale_waveforms
function scale_waveforms(lib::EventLibrary, val::Number)
  lib = copy(initialize(lib))
  lib.waveforms = lib.waveforms * val
  return lib
end
function scale_waveforms(data::DLData, val::Number)
  return DLData([scale_waveforms(lib, val) for lib in data])
end

type NaNError <: Exception
    msg::AbstractString
end

export check_nan
function check_nan(lib::EventLibrary)
  wf = waveforms(lib)
  nan_indices = find(x->isnan(x), wf)
  if length(nan_indices) > 0
    throw(NaNError("At indices $nan_indices"))
  end
end
check_nan(data::DLData) = for lib in data check_nan(lib) end
