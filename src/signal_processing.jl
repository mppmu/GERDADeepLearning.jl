# This file is a part of GERDADeepLearning.jl, licensed under the MIT License (MIT).

using DSP


export preprocess
function preprocess(env::DLEnv, sets::Dict{Symbol,EventLibrary}; steps_name="preprocessing", copyf=deepcopy)
  steps = convert(Array{String}, env.config[steps_name])
  result = copyf(sets)
  for (key, events) in result
    setname!(events, name(events)*"_preprocessed")
    put_label!(events, :FailedPreprocessing, zeros(Float32, length(events)))
    events.prop[:preprocessing] = steps
  end

  # perform the steps
  for (i,step) in enumerate(steps)
    info("Preprocesing $step.")
    pfunction = eval(parse(step))
    for (key, events) in result
      result[key] = pfunction(events)
    end
  end

  for (key, events) in result
    failed_count = length(find(f -> f != 0, events[:FailedPreprocessing]))
    if failed_count > 0
      info("Preprocesing failed for $failed_count events in $key. These have been tagged with the label :FailedPreprocessing = 1")
    end
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
  for i in 1:length(events)
    bl_level = dot(events.waveforms[1:bl_size, i], weights)
    events.waveforms[:, i] = events.waveforms[:, i] - bl_level
  end
  return events
end

export HE
function HE(events::EventLibrary)
  return filter(events, :E, E -> E>1500)
end

export align_peaks
function align_peaks(events::EventLibrary; target_length=256)
  currents = current_pulses(events; create_new=true)

  s = sample_size(events)
  half = Int64(target_length/2)
  rwf = zeros(Float32, target_length, length(events))

  for i in 1:length(events)
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

export align_centers
function align_midpoints(events::EventLibrary; center_y=0.5, target_length=256)
  charges = charge_pulses(events; create_new=true)

  s = sample_size(events)
  half = Int64(target_length/2)
  rwf = zeros(Float32, target_length, length(events))

  for i in 1:length(events)
    index = findmin(abs(charges.waveforms[:,i] - center_y))[2]
    if (index < half) || (index > s - half)
      events[:FailedPreprocessing][i] = 1
    else
      rwf[:,i] = events.waveforms[(index-half+1) : (index+half) , i]
    end
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

  for i in 1:length(events)
    top_level = dot(charges.waveforms[(end-top_size+1) : end, i], weights)
    events.waveforms[:,i] *= value / top_level
  end
  return events
end

export integrate
function integrate(events::EventLibrary)
  for i in 1:length(events)
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
  for i in 1:size(events.waveforms,2)
      events.waveforms[:,i] = gradient(events.waveforms[:,i])
  end
  if events[:waveform_type] == "current"
    events.prop[:waveform_type] = "other"
  elseif events[:waveform_type] == "charge"
    events.prop[:waveform_type] = "current"
  end
  return events
end
