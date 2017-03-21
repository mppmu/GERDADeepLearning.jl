# This file is a part of GERDADeepLearning.jl, licensed under the MIT License (MIT).


export FileKey
type FileKey
  run::Integer
  event_type::Symbol # :cal, :phy
  fileid::AbstractString

  function FileKey(id::AbstractString)
    if id[1:6] != "gerda-"
      throw(ArgumentError("$id is not a valid file identifier"))
    end
    run = parse(Int, id[10:13])
    event_type = Symbol(id[32:34])
    fileid = id[15:30]
    return new(run, event_type, fileid)
  end
end

export path
function path(base_dir::AbstractString, key::FileKey, tier)
  subfolder = (tier == :tier1 || tier == :tier2 || tier == :tierW) ? "ged" : "all"
  str_tier = string(tier)
  str_run = "run"*lpad(key.run, 4, 0)
  str_event_type = string(key.event_type)
  filename = "gerda-$str_run-$(key.fileid)-$str_event_type-$subfolder-$str_tier.root"
  return joinpath(base_dir, str_tier, subfolder, str_event_type, str_run, filename)
end

export parse_keylist
function parse_keylist(keylist_file::AbstractString)
  return [FileKey(id) for id in readlines(open(keylist_file))]
end
