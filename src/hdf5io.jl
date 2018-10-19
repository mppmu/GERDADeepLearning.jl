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

"""
The keylist must be iterable and implement listname.
The entries must support the string method.
"""
function create_extendible_hdf5_files(output_dir, keylists, detector_names, sample_size, chunk_size, label_keys)

  sample_size = Int64(sample_size) # Prevents HDF5 error
  chunk_size = Int64(chunk_size)

  h5files = [h5open(joinpath(output_dir, "$dname.h5"), "w") for dname in detector_names]

  label_arrays = Dict[]

  for (i,h5file) in enumerate(h5files)
    libroot = create_libroot(h5file, detector_names[i])

    detector_labels = Dict()

    waveforms = d_create(libroot, "waveforms", Float32, ((sample_size,chunk_size), (sample_size,-1)), "chunk", (sample_size,chunk_size)) # requires Int64 sizes
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
      attrs(klg)["name"] = listname(keylist)
      attrs(klg)["entries"] = [string(key) for key in keylist]
    end
  end

  return h5files, label_arrays
end


function lazy_read_all(dir::AbstractString)
  files = readdir(dir)
  files = filter(f->endswith(f, ".h5"), files)
  names = [file[1:end-3] for file in files]
  libs = [lazy_read_library(joinpath(dir, files[i]), names[i]) for i in 1:length(files)]
  result = DLData(libs)
  result.dir = dir
  return result
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
      info("Illegal waveform data for lib $(lib[:name]).")
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
    filepath = joinpath(dir, lib[:name]*".h5")
    write_lib(lib, filepath, uninitialize)
  end
end

export write_all
function write_all_multithreaded(data::DLData, dir::AbstractString, uninitialize::Bool)
  isdir(dir) || mkdir(dir)
  @everythread begin
    for i in threadpartition(1:length(data))
      lib = data.entries[i]
        filepath = joinpath(dir, lib[:name]*".h5")
        write_lib(lib, filepath, uninitialize)
    end
  end
end

export write_lib
function write_lib(lib::EventLibrary, filepath::AbstractString, uninitialize::Bool)
  h5open(filepath, "w") do h5file
    libroot = create_libroot(h5file, lib[:name])

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
    lib.initialization_function = lib2 -> _initialize_from_file(lib2, filepath, lib[:name])
  end
end



function _str_to_sym_dict(dict)
  result = Dict{Symbol, Any}()
  for (key,value) in dict
    result[Symbol(key)] = value
  end
  return result
end



function segh5_to_hdf5(formattype::AbstractString, keylists::Vector{Vector{AbstractString}}, output_dir::AbstractString, verbosity::Integer)
  all_files = vcat(keylists...)

  sample_size = _segh5_get_waveform_length(all_files[1])
  verbosity > 2 && info("Determined number of samples per waveform = $sample_size.")

  label_keys = [:keylist=>Int32, :filenum=>Int32, :E=>Float32, :E1=>Float32, :E2=>Float32, :E3=>Float32, :E4=>Float32]

  # Create HDF5 extendible storage
  h5files, h5_arrays = create_extendible_hdf5_files(output_dir, [], [formattype], sample_size, 256, label_keys)
  out_keylist = h5_arrays[1][:keylist]
  out_filenum = h5_arrays[1][:filenum]
  out_E = h5_arrays[1][:E]
  out_E1 = h5_arrays[1][:E1]
  out_E2 = h5_arrays[1][:E2]
  out_E3 = h5_arrays[1][:E3]
  out_E4 = h5_arrays[1][:E4]
  out_waveforms = h5_arrays[1][:waveforms]

  try
      file_count = length(all_files)
      file_i = 0
      for(kl_i, filelist) in enumerate(keylists)
          for file in filelist
            if !isfile(file)
              verbosity >= 1 && info("Skipping entry because file does not exist: $file")
            else
              verbosity >= 2 && info("Reading file $file_i / $file_count...")
              verbosity >= 3 && info("File: $file")

              h5in = h5open(file)

              # println(size(out_waveforms))
              # println(size(h5in["DAQ_Data"]["daq_pulses"]))
              # flush(STDOUT)

              f_energies = read(h5in["Processed_data"]["energies"])
              f_pulses = h5in["DAQ_Data"]["daq_pulses"][1,:,:][1,:,:]

              extend_h5_array(out_keylist, fill(kl_i, size(f_energies,2)))
              extend_h5_array(out_filenum, fill(file_i, size(f_energies,2)))

              extend_h5_array(out_E, f_energies[1,:])
              extend_h5_array(out_E1, f_energies[2,:])
              extend_h5_array(out_E2, f_energies[3,:])
              extend_h5_array(out_E3, f_energies[4,:])
              extend_h5_array(out_E4, f_energies[5,:])

              extend_h5_matrix(out_waveforms, f_pulses)

              out_E
            end
            file_i += 1
          end
      end
  catch exc
      #verbosity >= 1 && info("Fatal error during conversion to HDF5. Closing HDF5 files and rethrowing error.")
      for h5file in h5files close(h5file) end
      rethrow(exc)
  end

verbosity >= 3 && info("Finished conversion without fatal errors. Closing HDF5 files...")

for h5file in h5files close(h5file) end

verbosity >= 3 && info("All HDF5 files closed.")
end


function _segh5_get_waveform_length(filepath)
  file = h5open(filepath)
  return size(file["DAQ_Data"]["daq_pulses"])[2] # #channels, #samples, #events
end


function extend_h5_array(array::HDF5Dataset, newdata::Vector)
  set_dims!(array, (length(array)+length(newdata),) )
  array[(end-length(newdata)+1) : end] = newdata
end

function extend_h5_matrix(matrix::HDF5Dataset, newdata::Matrix)
  set_dims!(matrix, (size(matrix,1), size(matrix,2)+size(newdata,2)))
  matrix[:, (end-size(newdata, 2)+1) : end] = newdata
end
