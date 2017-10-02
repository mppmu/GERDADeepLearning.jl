# This file is a part of GERDADeepLearning.jl, licensed under the MIT License (MIT).

using ROOTFramework, Cxx, MGDO, Base.Threads, MultiThreadingTools, HDF5


function mgdo_to_hdf5(base_path::AbstractString, output_dir::AbstractString, keylists::Vector{KeyList}; sample_size=1000, verbosity=2)

  label_keys = [:keylist=>Int8, :timestamp=>UInt64, :E=>Float32, :AoE=>Float32, :AoE_class=>Int8, :ANN_mse_class=>Int8, :ANN_alpha_class=>Int8, :isTP=>Int8, :isBL=>Int8, :multiplicity=>Int32, :isMuVetoed=>Int8, :isLArVetoed=>Int8]

  # Create HDF5 extendible storage
  h5files, h5_arrays = create_extendible_hdf5_files(output_dir, keylists, phase2_detectors, sample_size, 256, label_keys)

  try
    file_count = length(merge(keylists...))
    total_file_i = 0
    for (keylist_id, keylist) in enumerate(keylists)
      for (file_i,filekey) in enumerate(keylist.entries)
        total_file_i += 1
        file1 = path(base_path, filekey, :tier1)
        file4 = path(base_path, filekey, :tier4)
        if !isfile(file1)
          verbosity >= 1 && info("Skipping entry because tier1 file does not exist: $file1")
        elseif !isfile(file4)
            verbosity >= 1 && info("Skipping entry because tier4 file does not exist: $file4")
        else
          verbosity >= 2 && info("Reading file $total_file_i / $file_count...")
          verbosity >= 3 && info("Tier1: $file1")
          verbosity >= 3 && info("Tier4: $file4")
          is_physics_data = filekey.event_type == :phy
          results = ThreadLocal{Any}()
          # Run every thread over part of the data
          @everythread begin
            thread_result = read_single_thread_single_file(phase2_detectors, file1, file4, label_keys, verbosity, sample_size, keylist_id, is_physics_data)
            results[] = thread_result
          end

          # Write parts to HDF5 files
          results = merge_labels_per_detector(all_thread_values(results), label_keys)
          append_to_hdf5(results, h5_arrays, verbosity)
        end
      end
    end
  catch exc
    verbosity >= 1 && info("Fatal error during conversion to HDF5. Closing HDF5 files and rethrowing error.")
    for h5file in h5files close(h5file) end
    rethrow(exc)
  end
    
  verbosity >= 3 && info("Finished conversion without fatal errors. Closing HDF5 files...")

  for h5file in h5files close(h5file) end
    
  verbosity >= 3 && info("All HDF5 files closed.")
end


function read_single_thread_single_file(detector_names, file1, file4, label_keys, verbosity, sample_size, keylist_id, is_physics_data::Bool)
  # Create thread-local arrays
  tier4_bindings = TTreeBindings()
  branch_timestamp = tier4_bindings[:timestamp] = Ref(zero(UInt64))
  branch_energies = tier4_bindings[:energy] = zeros(Float64, 0)
  branch_event_ch = tier4_bindings[:eventChannelNumber] = Ref(zero(Int32))
  # branch_timestamp = tier4_bindings[:timestamp] = Ref(zero(UInt64))
  branch_aoeVeto = tier4_bindings[:isAoEvetoed] = zeros(Int32, 0)
  branch_aoeEval = tier4_bindings[:isAoEevaluated] = Bool[]
  branch_aoeVal = tier4_bindings[:AoEclassifier] = zeros(Float64, 0)
  branch_isTP = tier4_bindings[:isTP] = Ref(zero(Int32))
  branch_isBL = tier4_bindings[:isBL] = Ref(zero(Int32))
  branch_multiplicity = tier4_bindings[:multiplicity] = Ref(zero(Int32))
  branch_isMuVetoed = tier4_bindings[:isMuVetoed] = Ref(zero(Int32))
  branch_ANN_mse_eval = tier4_bindings[:psdIsEval_ANN_mse] = Bool[]
  branch_ANN_mse_flag = tier4_bindings[:psdFlag_ANN_mse] = zeros(Int32, 0)
  branch_ANN_alpha_eval = tier4_bindings[:psdIsEval_ANN_alpha] = Bool[]
  branch_ANN_alpha_flag = tier4_bindings[:psdFlag_ANN_alpha] = zeros(Int32, 0)
  if is_physics_data
    branch_isLArVetoed = tier4_bindings[:isLArVetoed] = Ref(zero(Int32))
  end

  # Prepare tables
  result, keylist, timestamp, E, AoE, AoE_class, ANN_mse_class, ANN_alpha_class, isTP, isBL, multiplicity, isMuVetoed, isLArVetoed = create_label_arrays(label_keys, length(detector_names))
  waveforms = [Vector{Float32}[] for i in 1:length(detector_names)]
  result[:waveforms] = waveforms

  # Open tree and read events
  open(MGTEventTree{JlMGTEvent}, file1) do tier1_tree
    open(TChainInput, tier4_bindings, "tier4", file4) do tier4_input

      assert(length(tier1_tree) == length(tier4_input))
      n = length(tier1_tree)

      for i in threadpartition(eachindex(tier1_tree))
        t1_evt = tier1_tree[i]
        tier4_input[i]

        for (no_in_event, detector_i) in enumerate(t1_evt.waveforms.ch) # 0:39
          detector = detector_i + 1
                    
          if branch_energies[detector] > 0
              push!(waveforms[detector], convert(Array{Float32}, t1_evt.aux_waveforms.samples[no_in_event]))

              push!(keylist[detector], keylist_id)
              push!(timestamp[detector], branch_timestamp.x)
              push!(E[detector], branch_energies[detector])

              push!(isTP[detector], branch_isTP.x)
              push!(isBL[detector], branch_isBL.x)
              push!(multiplicity[detector], branch_multiplicity.x)
              push!(isMuVetoed[detector], branch_isMuVetoed.x)
              if is_physics_data
                push!(isLArVetoed[detector], branch_isLArVetoed.x)
              else
                push!(isLArVetoed[detector], 0)
              end

              # A/E
              push!(AoE[detector], branch_aoeVal[detector])
              if branch_aoeEval[detector]
                  push!(AoE_class[detector], branch_aoeVeto[detector])
              else
                  push!(AoE_class[detector], -1)
              end
              # ANN - MSE
              if branch_ANN_mse_eval[detector]
                push!(ANN_mse_class[detector], branch_ANN_mse_flag[detector])
              else
                push!(ANN_mse_class[detector], -1)
              end
              # ANN - Alpha
              if branch_ANN_alpha_eval[detector]
                push!(ANN_alpha_class[detector], branch_ANN_alpha_flag[detector])
              else
                push!(ANN_alpha_class[detector], -1)
              end
           end # if E
        end # for
      end # for
    end # open
  end # open

  return result
end

function create_label_arrays(label_keys, detector_count)
  list = Vector{Vector}[]
  d = Dict{Symbol, Vector}()
  for (key, dtype) in label_keys
    a = [dtype[] for i in 1:detector_count]
    d[key] = a
    push!(list, a)
  end
  return d, list...
end

function merge_labels_per_detector(events_list, #::Vector{Dict{Symbol,Vector}}
  label_keys)

  detector_count = length(events_list[1][:waveforms])
  total_event_count = sum([length(events_list[i][:waveforms][1]) for i in 1:length(events_list)])

  result, arrays = create_label_arrays(label_keys, detector_count)
  result[:waveforms] = fill(Vector{Float32}[], detector_count)

  for detector in 1:detector_count
    for label_key in keys(result)
      # cat arrays belonging to the same type and detector
      arrays = [events_list[i][label_key][detector] for i in 1:length(events_list)]
      combined_label = vcat(arrays...)
      result[label_key][detector] = combined_label
    end
  end

  return result
end



"""
h5_arrays :: Detector -> Type
results :: Type -> Detector
"""
function append_to_hdf5(results, h5_arrays, verbosity)
  verbosity >= 3 && info("Appending results to HDF5 files: $(length(results)) categories with entries per detector: $(length.(results[:E]))")
    verbosity >= 4 && info("Energy arrays of detectors 1 and 2 start with: $(results[:E][1][1:10]) (total length $(length(results[:E][1]))) and $(results[:E][2][1:10]) (total length $(length(results[:E][2])))")
  for detector in 1:length(h5_arrays)
    for key in keys(results)
      h5_array = h5_arrays[detector][key]
      new_array = results[key][detector]

      if length(new_array) > 0
        if length(size(h5_array)) == 1
          set_dims!(h5_array, (length(h5_array)+length(new_array),))
          h5_array[(end-length(new_array)+1) : end] = new_array
        elseif length(size(h5_array)) == 2
          set_dims!(h5_array, (size(h5_array,1), size(h5_array,2)+length(new_array)))
          new_array_2D = hcat(new_array...)
          h5_array[:, (end-length(new_array)+1) : end] = new_array_2D
        else
          throw(ArgumentError("more than 2D array encountered"))
        end
      end
    end
  end
end


"""
Create one HDF5 file for each keylist & detector.
Datasets are split when reading the HDF5.
"""
function read_single_tier_1_4(
  file1, file4,
  h5files,
  verbosity,
  file_i,
  file_count
  )
  #export JULIA_NUM_THREADS=8
  # Run with X threads over one file
  #@everythread threadsafe_info("threadid(): $(threadid()), my fraction: $(threadfraction(1:100))")

  tier4_bindings = TTreeBindings()
  energies = tier4_bindings[:energy] = zeros(Float64, 0)
  event_ch = tier4_bindings[:eventChannelNumber] = Ref(zero(Int32))
  # timestamp = tier4_bindings[:timestamp] = Ref(zero(UInt64))
  aoeVeto = tier4_bindings[:psdFlag_AoE] = zeros(Int32, 0)
  aoeEval = tier4_bindings[:psdIsEval_AoE] = Bool[]
  aoeVal = tier4_bindings[:psdClassifier_AoE] = zeros(Float64, 0)
  isTP = tier4_bindings[:isTP] = Ref(zero(Int32))
  isBL = tier4_bindings[:isBL] = Ref(zero(Int32))

  # Prepare tables
  result = Dict{Symbol, EventLibrary}()
  waveform_map = Vector{Vector{Float32}}[]
  E_map = Vector{Float32}[]
  AoE_map = Vector{Float32}[]
  AoE_class_map = Vector{Float32}[]
  isTP_map = Vector{Float32}[]
  isBL_map = Vector{Float32}[]

  for (i,set_name) in enumerate(set_names)
    events = EventLibrary(zeros(Float32, 0, 0))
    E_map[i] = events.labels[:E] = Float32[]
    AoE_map[i] = events.labels[:AoE] = Float32[]
    AoE_class_map[i] = events.labels[:AoE_class] = Float32[]
    isTP_map[i] = events.labels[:isTP] = Float32[]
    isBL_map[i] = events.labels[:isBL] = Float32[]
    events.prop[:name] = string(set_name)
    events.prop[:waveform_type] = "raw"
    result[set_name] = events
    waveform_map[i] = Array{Float32}[]
  end

  @everythread begin

    tier4_tchain = TChain("tier4", file4)
    open(MGTEventTree{JlMGTEvent}, file1) do tier1_tree

      assert(length(tier1_tree) == length(tier4_tchain))
      n = length(tier1_tree)
      verbosity >= 2 && info("Reading file $file_i / $file_count. Entries: $n")

      tier4_input = TTreeInput(tier4_tchain, tier4_bindings)

      switch_indices = setsplit(set_sizes[file_i], n)
      set_index = 1

      for (t1_evt, i) in zip(tier1_tree, tier4_input)
        for detector_i in select_channels
          detector = detector_i + 1
          wf_index = findfirst(detector_i in t1_evt.waveforms.ch)
          if wf_index > 0
            push!(waveform_map[set_index], convert(Array{Float32}, t1_evt.aux_waveforms.samples[wf_index]))

            push!(E_map[set_index], energies[detector])
            push!(AoE_map[set_index], aoeVal[detector])
            if aoeEval[detector]
                push!(AoE_class_map[set_index], aoeVeto[detector])
            else
                push!(AoE_class_map[set_index], -1)
            end
            push!(isBL_map[set_index], isBL.x)
            push!(isTP_map[set_index], isTP.x)
          end
        end

        # switch sets
        if i in switch_indices
          set_index += 1
        end
      end
    end

  end


  # Convert waveforms to 2D array
  for set_index in 1:length(set_sizes)
    if length(waveform_map[set_index]) > 0
      tab_waveforms = hcat(waveform_map[set_index]...)
    else
      tab_waveforms = zeros(Float32, 0,0)
    end
    result[set_names[set_index]].waveforms = tab_waveforms
  end

  return result
end


function read_tiers_1_4(
  base_path::AbstractString,
  files::Array{FileKey};
  set_names=[:data],
  set_sizes=fill([1], length(files)),
  select_channels=0:39,
  verbosity = 2
  )



  for (file_i,filekey) in enumerate(files)
    file1 = path(base_path, filekey, :tier1)
    file4 = path(base_path, filekey, :tier4)
    if !isfile(file1)
      verbosity >= 1 && info("Skipping entry because tier1 file does not exist: $file1")
    elseif !isfile(file4)
        verbosity >= 1 && info("Skipping entry because tier4 file does not exist: $file4")
    else
      tier4_tchain = TChain("tier4", file4)
      tier1_tree = open(MGTEventTree{JlMGTEvent}, file1)

      datasets = read_single_tier_1_4(file1, file4, set_names, set_sizes, select_channels, verbosity, file_i, length(files))
    end
  end

end

# function read_tiers_1_4(
#   base_path::AbstractString,
#   files::Array{FileKey};
#   set_names=[:data],
#   set_sizes=fill([1], length(files)),
#   select_channels=0:39,
#   log_progress=true,
#   log_errors=true
#   )
#
#   tier4_bindings = TTreeBindings()
#   energies = tier4_bindings[:energy] = zeros(Float64, 0)
#   event_ch = tier4_bindings[:eventChannelNumber] = Ref(zero(Int32))
#   # timestamp = tier4_bindings[:timestamp] = Ref(zero(UInt64))
#   aoeVeto = tier4_bindings[:psdFlag_AoE] = zeros(Int32, 0)
#   aoeEval = tier4_bindings[:psdIsEval_AoE] = Bool[]
#   aoeVal = tier4_bindings[:psdClassifier_AoE] = zeros(Float64, 0)
#   isTP = tier4_bindings[:isTP] = Ref(zero(Int32))
#   isBL = tier4_bindings[:isBL] = Ref(zero(Int32))
#
#   # Prepare tables
#   result = Dict{Symbol, EventLibrary}()
#   waveform_lists::Dict{Symbol, Array{Array{Float32}}} = Dict()
#   for (i,set_name) in enumerate(set_names)
#     events = EventLibrary(zeros(Float32, 0, 0))
#     events.labels[:E] = Float32[]
#     events.labels[:AoE] = Float32[]
#     events.labels[:AoE_class] = Float32[]
#     events.labels[:isTP] = Float32[]
#     events.labels[:isBL] = Float32[]
#     events.prop[:name] = string(set_name)
#     events.prop[:waveform_type] = "raw"
#     result[set_name] = events
#     waveform_lists[set_name] = Array{Float32}[]
#   end
#
#
#   for (file_i,filekey) in enumerate(files)
#     file1 = path(base_path, filekey, :tier1)
#     file4 = path(base_path, filekey, :tier4)
#     if !isfile(file1)
#       log_errors && info("Skipping entry because tier1 file does not exist: $file1")
#     elseif !isfile(file4)
#         log_errors && info("Skipping entry because tier4 file does not exist: $file4")
#     else
#       tier4_tchain = TChain("tier4", file4)
#       tier1_tree = open(MGTEventTree{JlMGTEvent}, file1)
#
#       assert(length(tier1_tree) == length(tier4_tchain))
#       n = length(tier1_tree)
#       log_progress && info("Reading file $file_i / $(length(files)). Entries: $n")
#
#       tier4_input = TTreeInput(tier4_tchain, tier4_bindings)
#
#       switchdict = setsplit(set_names, set_sizes[file_i], n)
#
#       tab_energies, tab_aoeValues, tab_aoeClasses, list_waveforms, tab_baselines, tab_testpulses = lookup(result, waveform_lists, set_names[1])
#
#       for (t1_evt, i) in zip(tier1_tree, tier4_input)
#           for detector_i in select_channels
#             detector = detector_i + 1
#             wf_index = findfirst(detector_i in t1_evt.waveforms.ch)
#             if wf_index > 0
#               push!(list_waveforms, convert(Array{Float32}, t1_evt.aux_waveforms.samples[wf_index]))
#
#               push!(tab_energies, energies[detector])
#               push!(tab_aoeValues, aoeVal[detector])
#               if aoeEval[detector]
#                   push!(tab_aoeClasses, aoeVeto[detector])
#               else
#                   push!(tab_aoeClasses, -1)
#               end
#               push!(tab_baselines, isBL.x)
#               push!(tab_testpulses, isTP.x)
#             end
#           end
#
#           # switch sets
#           if haskey(switchdict, i[1])
#             nextset = switchdict[i[1]]
#             tab_energies, tab_aoeValues, tab_aoeClasses, list_waveforms, tab_baselines, tab_testpulses = lookup(result, waveform_lists, nextset)
#           end
#         end
#       end
#     end
#
#   # Convert waveforms to 2D array
#   for set_name in set_names
#     if length(waveform_lists[set_name]) > 0
#       tab_waveforms = hcat(waveform_lists[set_name]...)
#     else
#       tab_waveforms = zeros(Float32, 0,0)
#     end
#     result[set_name].waveforms = tab_waveforms
#   end
#
#   return result
# end


# export read_tier_W
# function read_tier_W(
#   base_path::AbstractString,
#   files::Array{FileKey};
#   set_names=[:training, :test],
#   set_sizes=fill([0.7, 0.3], length(files)),
#   select_channels=0:39
#   )
#
#   tierW_bindings = TTreeBindings()
#   evtNo = tierW_bindings[Symbol("tier2.WaveletCoeffs_evtNo")] = zeros(Int32, 0)
#   channels = tierW_bindings[Symbol("tier2.WaveletCoeffs_ch")] = zeros(Int32, 0)
#
#   # std:vector containing wavelet coefficients
#   # IMPORTANT: indexing on std:vector uses c-style integers 0 to len-1
#   samples_cxx = tierW_bindings[Symbol("tier2.WaveletCoeffs_samples")] =
#       ROOTFramework.CxxObjWithPtrRef(icxx"std::vector<std::vector<double>>();")
#
#   # tier2.WaveletCoeffs_wlCoeffs, tier2.WaveletCoeffs_samples
#
#   tier4_bindings = TTreeBindings()
#   energies = tier4_bindings[:energy] = zeros(Float64, 0)
#   event_ch = tier4_bindings[:eventChannelNumber] = Ref(zero(Int32))
#   # timestamp = tier4_bindings[:timestamp] = Ref(zero(UInt64))
#   aoeVeto = tier4_bindings[:psdFlag_AoE] = zeros(Int32, 0)
#   aoeEval = tier4_bindings[:psdIsEval_AoE] = Bool[]
#   aoeVal = tier4_bindings[:psdClassifier_AoE] = zeros(Float64, 0)
#   samples = Float64[]
#
#   # Prepare tables
#   result = Dict{Symbol, EventLibrary}()
#   waveform_lists::Dict{Symbol, Array{Array{Float32}}} = Dict()
#   for (i,set_name) in enumerate(set_names)
#     events = EventLibrary(zeros(Float32, 0, 0))
#     events.labels[:E] = Float32[]
#     events.labels[:AoE] = Float32[]
#     events.labels[:AoE_class] = Float32[]
#     events.prop[:name] = string(set_name)
#     events.prop[:waveform_type] = "current"
#     result[set_name] = events
#     waveform_lists[set_name] = Array{Float32}[]
#   end
#
#
#   @time for (file_i,filekey) in enumerate(files)
#     fileW = path(base_path, filekey, :tierW)
#     file4 = path(base_path, filekey, :tier4)
#     if !isfile(fileW) || !isfile(file4)
#       info("Skipping because files don't exist: $fileW")
#     else
#       tierW_tchain = TChain("tree", fileW)
#       tier4_tchain = TChain("tier4", file4)
#
#       assert(length(tierW_tchain) == length(tier4_tchain))
#       n = length(tierW_tchain)
#       info("Reading file $file_i / $(length(files)). Entries: $n")
#
#       tierW_input = TTreeInput(tierW_tchain, tierW_bindings)
#       tier4_input = TTreeInput(tier4_tchain, tier4_bindings)
#
#       switchdict = setsplit(set_names, set_sizes[file_i], n)
#
#       tab_energies, tab_aoeValues, tab_aoeClasses, list_waveforms = lookup(result, waveform_lists, set_names[1])
#
#       for i in zip(tierW_input, tier4_input)
#           for detector_i in select_channels
#                 detector = detector_i + 1
#             if length(samples_cxx.x[detector-1]) > 0
#               resize!(samples, length(samples_cxx.x[detector-1]))
#               copy!(samples, samples_cxx.x[detector-1])
#
#               # Add new values to lists and arrays
#               push!(tab_energies, energies[detector])
#               push!(tab_aoeValues, aoeVal[detector])
#               push!(list_waveforms, convert(Array{Float32}, samples))
#               if aoeEval[detector]
#                   push!(tab_aoeClasses, aoeVeto[detector])
#               else
#                   push!(tab_aoeClasses, -1)
#               end
#             end
#           end
#
#           # switch sets
#           if haskey(switchdict, i[1])
#             nextset = switchdict[i[1]]
#             tab_energies, tab_aoeValues, tab_aoeClasses, list_waveforms = lookup(result, waveform_lists, nextset)
#           end
#         end
#       end
#     end
#
#   for set_name in set_names
#     if length(waveform_lists[set_name]) > 0
#       tab_waveforms = hcat(waveform_lists[set_name]...)
#     else
#       tab_waveforms = zeros(Float32, 0,0)
#     end
#     result[set_name].waveforms = tab_waveforms
#   end
#
#   return result
# end

function setsplit(set_names, set_sizes, n)
  sizes = convert(Array{Int},round(set_sizes * n))
  result = Dict()
  total = 0
  for (i,size) in enumerate(sizes)
    if i != 1
      result[total] = set_names[i]
    end
    total += size
  end
  return result
end

function setsplit(set_sizes, n)
  sizes = convert(Array{Int},round(set_sizes * n))
  result = Int64[]
  total = 0
  for (i,size) in enumerate(sizes)
    if i != 1
      push!(result, total)
    end
    total += size
  end
  return result
end

function lookup(result, waveform_lists, set_name)
  tab_energies = result[set_name].labels[:E]
  tab_aoeValues = result[set_name].labels[:AoE]
  tab_aoeClasses = result[set_name].labels[:AoE_class]
  list_waveforms = waveform_lists[set_name]
  tab_baselines = result[set_name].labels[:isBL]
  tab_testpulses = result[set_name].labels[:isTP]
  return tab_energies, tab_aoeValues, tab_aoeClasses, list_waveforms, tab_baselines, tab_testpulses
end


# ! load MGDO
# export read_tier_1
# function read_tier_1(files)
#   tier1 = TTreeBindings()
#   fAuxWaveforms = tier1[Symbol("MGTree.event.fAuxWaveforms")] =
#       ROOTFramework.CxxObjWithPtrRef(icxx"std::vector<std::vector<double>>();")
#   fUniqueID = tier1[Symbol("MGTree.event.fUniqueID")] = Ref(zero(Int64))
#
#   tmpF64 = Float64[]
#   waveforms = Array{Float32}[]
#
#   for file in files
#     info("Reading file $file")
#     chain1 = TChain("MGTree", file)
#     data = TTreeInput(chain1, tier1)
#     n = @cxx chain1->GetEntries()
#     info("Tree has length $n")
#
#     for i in data
#       info(i)
#       for detector_i in linearindices(fAuxWaveforms.x) # c-style indices 0:39
#         detector = detector_i + 1
#         if length(fAuxWaveforms.x[detector-1]) > 0
#             # read waveform
#             resize!(tmpF64, length(fAuxWaveforms.x[detector-1]))
#             copy!(tmpF64, fAuxWaveforms.x[detector-1])
#             push!(waveforms, convert(Array{Float32}, tmpF64))
#           end
#       end
#     end
#   end
#   tab_waveforms = hcat(waveforms...)
#   return Dict(:waveforms => tab_waveforms)
# end
#
# export resolve_files
# function resolve_files(path::AbstractString, keylists::Array{AbstractString})
#
# end
