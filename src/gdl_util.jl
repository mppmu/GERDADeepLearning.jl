# This file is a part of GERDADeepLearning.jl, licensed under the MIT License (MIT).


export mapvalues
function mapvalues(d::Dict, f, post_params...)
  result = copy(d)
  for (key,value) in d
    result[key] = f(value, post_params...)
  end
  return result
end

function mapvalues(data::DLData, f, post_params...)
  return DLData([f(lib, post_params...) for lib in data])
end
