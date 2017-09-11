# This file is a part of GERDADeepLearning.jl, licensed under the MIT License (MIT).

using HDF5, Base.Threads, MultiThreadingTools

type EventFormatException <: Exception
  msg::String
end


function create_libroot(h5file, libname)
  libroot = g_create(h5file, libname)
  attrs(libroot)["type"] = "EventLibrary"
  attrs(libroot)["version"] = "1.0"
  return libroot
end

function create_extendible_hdf5_files(output_dir, keylists, detector_names, sample_size, chunk_size, label_keys)
  h5files = [h5open(joinpath(output_dir, "$dname.h5"), "w") for dname in detector_names]

  label_arrays = Dict[]

  for (i,h5file) in enumerate(h5files)
    libroot = create_libroot(h5file, detector_names[i])

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


function lazy_read_all(dir::AbstractString)
  files = readdir(dir)
  names = [file[1:end-3] for file in files]
  libs = [lazy_read_library(joinpath(dir, files[i]), names[i]) for i in 1:length(files)]
  return DLData(libs)
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
  lib.prop[:eventcount] = size(libroot["waveforms"], 2)
  # TODO keylists
                    
  # Initialize label names
  labels = libroot["labels"]
  for key in names(labels)
    lib.labels[Symbol(key)] = zeros(Float32, 0)
  end

  close(ifile)

  return lib
end

function _initialize_from_file(lib::EventLibrary, h5_filepath, libname)
  h5open(h5_filepath, "r") do ifile
    libroot = ifile[libname]

    # Read waveforms
    try
      lib.waveforms = read(libroot["waveforms"])
    catch err
      info("Illegal waveform data for lib $(name(lib)).")
      lib.waveforms = zeros(Float32, 1, 0)
    end

    # Read labels
    labels = libroot["labels"]
    for key in names(labels)
      data = read(labels[key])
      lib.labels[Symbol(key)] = data
    end
  end
end


export write_all
function write_all_sequentially(data::DLData, dir::AbstractString, uninitialize::Bool)
  isdir(dir) || mkdir(dir)
  for lib in data
    filepath = joinpath(dir, name(lib)*".h5")
    write_lib(lib, filepath, uninitialize)
  end
end

export write_all
function write_all_multithreaded(data::DLData, dir::AbstractString, uninitialize::Bool)
  isdir(dir) || mkdir(dir)
  @everythread begin
    for i in threadpartition(1:length(data))
      lib = data.entries[i]
        filepath = joinpath(dir, name(lib)*".h5")
        write_lib(lib, filepath, uninitialize)
    end
  end
end

export write_lib
function write_lib(lib::EventLibrary, filepath::AbstractString, uninitialize::Bool)
  h5open(filepath, "w") do h5file
    libroot = create_libroot(h5file, name(lib))

    write(libroot, "waveforms", waveforms(lib))

    labelsg = g_create(libroot, "labels")
    for (key,value) in lib.labels
      write(labelsg, string(key), value)
    end

    properties = g_create(libroot, "properties")
    for(key,value) in lib.prop
      try
        attrs(properties)[string(key)] = value
      catch err
        threadsafe_info("Cannot write property $key: $err")
      end
    end
  end

  if uninitialize
    dispose(lib)
    lib.initialization_function = lib2 -> _initialize_from_file(lib2, filepath, name(lib))
  end
end



function _str_to_sym_dict(dict)
  result = Dict{Symbol, Any}()
  for (key,value) in dict
    result[Symbol(key)] = value
  end
  return result
end
