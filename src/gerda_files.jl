# This file is a part of GERDADeepLearning.jl, licensed under the MIT License (MIT).


export FileKey
immutable FileKey
  run::Integer
  event_type::Symbol # :cal, :phy
  fileid::AbstractString
  id::AbstractString

  function FileKey(id::AbstractString)
    id = strip(id)
    if id[1:6] != "gerda-"
      throw(ArgumentError("$id is not a valid file identifier"))
    end
    run = parse(Int, id[10:13])
    event_type = Symbol(id[32:34])
    fileid = id[15:30]
    return new(run, event_type, fileid, id)
  end
end

function Base.string(key::FileKey)
  return key.id
end

export KeyList
immutable KeyList
  name::String
  entries::Vector{FileKey}
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
function parse_keylist(keylist_file::AbstractString, name::String)
  entries = [FileKey(id) for id in readlines(open(keylist_file))]
  return KeyList(name, entries)
end

export name
function name(keylist::KeyList)
  return keylist.name
end

function Base.merge(keylists::KeyList...)
  return KeyList("", vcat([kl.entries for kl in keylists]...))
end

function Base.length(keylist::KeyList)
  return length(keylist.entries)
end


phase2_detectors = ["GD91A", "GD35B", "GD02B", "GD00B", "GD61A", "GD89B", "GD02D", "GD91C", # string1
		"ANG5", "RG1", "ANG3", # string2
		"GD02A", "GD32B", "GD32A", "GD32C", "GD89C", "GD61C",  "GD76B", "GD00C", # string3
		"GD35C", "GD76C", "GD89D", "GD00D", "GD79C", "GD35A", "GD91B", "GD61B", # string4
		"ANG2", "RG2", "ANG4", # string 5
		"GD00A", "GD02C", "GD79B", "GD91D", "GD32D", "GD89A", "ANG1", # string6
		"GTF112", "GTF32", "GTF45_2" # string7
    ]

export get_detector_index
function get_detector_index(detector_name::String)
  return find(n->n==detector_name, phase2_detectors)[1] - 1
end


export parse_detectors
function parse_detectors(list::Vector)
  return [(isa(det,Integer) ? det : get_detector_index(det)) for det in list]
end
