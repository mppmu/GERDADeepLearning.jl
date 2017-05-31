# This file is a part of GERDADeepLearning.jl, licensed under the MIT License (MIT).

using HDF5

function create_extendible_hdf5_files(output_dir, keylists, detector_names, sample_size, chunk_size, label_keys)
  h5files = [h5open(joinpath(output_dir, "$dname.h5"), "w") for dname in detector_names]

  label_arrays = Dict[]

  for (i,h5file) in enumerate(h5files)
    libroot = g_create(h5file, detector_names[i])

    attrs(libroot)["type"] = "EventLibrary"
    attrs(libroot)["version"] = "1.0"

    detector_labels = Dict()

    waveforms = d_create(libroot, "waveforms", Float32, ((sample_size,chunk_size), (sample_size,-1)), "chunk", (sample_size,chunk_size))
    set_dims!(waveforms, (sample_size, 0))
    detector_labels[:waveforms] = waveforms

    labelsg = g_create(libroot, "labels")
    push!(label_arrays, detector_labels)
    for (label_key, dtype) in label_keys
      labeld = d_create(labelsg, "$label_key", dtype, ((chunk_size,), (-1,)), "chunk", (chunk_size,))
      set_dims!(labeld, (0,))
      detector_labels[label_key] = labeld
    end

    properties = g_create(libroot, "properties")
    attrs(properties)["name"] = detector_names[i]
    attrs(properties)["sampling_rate"] = 100e6
    attrs(properties)["detector_id"] = i
    attrs(properties)["detector_name"] = detector_names[i]
    attrs(properties)["waveform_type"] = "raw"
    for (i,keylist) in enumerate(keylists)
      klg = g_create(properties, "Keylist$i")
      attrs(klg)["name"] = name(keylist)
      attrs(klg)["entries"] = [string(key) for key in keylist.entries]
    end
  end

  return h5files, label_arrays
end

type EventFormatException <: Exception
  msg::String
end



function lazy_read_library(h5_filepath, libname)
  init = lib -> _initialize_from_file(lib, h5_filepath, libname)
  lib =  EventLibrary(init)

  ifile = h5open(h5_filepath, "r")
  libroot = ifile[libname]

  if read(attrs(libroot)["type"]) != "EventLibrary"
    throw(EventFormatException("Not a valid EventLibrary: $(read(attrs(libroot)["type"]))"))
  end
  if read(attrs(libroot)["version"]) != "1.0"
    throw(EventFormatException("Unknown EventLibrary version: $(read(attrs(libroot)["version"]))"))
  end

  # Read properties
  props = libroot["properties"]
  for key in names(attrs(props))
    value = read(attrs(props)[key])
    lib.prop[Symbol(key)] = value
  end
  # TODO keylists

  close(ifile)

  return lib
end

function _initialize_from_file(lib::EventLibrary, h5_filepath, libname)
  ifile = h5open(h5_filepath, "r")
  libroot = ifile[libname]

  # Read waveforms
  lib.waveforms = read(libroot["waveforms"])

  # Read labels
  labels = libroot["labels"]
  for key in names(labels)
    data = read(labels[key])
    lib.labels[Symbol(key)] = data
  end

  close(ifile)
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



function _str_to_sym_dict(dict)
  result = Dict{Symbol, Any}()
  for (key,value) in dict
    result[Symbol(key)] = value
  end
  return result
end
