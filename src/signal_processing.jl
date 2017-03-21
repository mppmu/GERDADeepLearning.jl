# This file is a part of GERDADeepLearning.jl, licensed under the MIT License (MIT).


export preprocess
function preprocess(env::DLEnv, sets::Dict{Symbol,EventLibrary})
  steps = convert(Array{String}, env.config["preprocessing"])
  result = copy(sets)
  for (name, events) in result
    events.prop[:name] = events[:name]*"_preprocessed"
    events.prop[:preprocessing] = steps
  end

  # perform the steps
  for step in steps
    pfunction = eval(parse(step))
    for (name, events) in result
      result[name] = pfunction(events)
    end
  end
  return result
end

# function preprocess(env::DLEnv, events::EventLibrary)
# end

export normalize_energy
function normalize_energy(events::EventLibrary)
  events = copy(events)
  s = size(events.waveforms, 1)
  for i in 1:size(events.waveforms,2)
    events.waveforms[:,i] = events.waveforms[:,i]*s / sum(events.waveforms[:,i])
  end
  return events
end
