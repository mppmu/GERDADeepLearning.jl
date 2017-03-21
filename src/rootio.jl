# This file is a part of GERDADeepLearning.jl, licensed under the MIT License (MIT).

using ROOTFramework, Cxx


export read_tier_W
function read_tier_W(
  base_path::AbstractString,
  files::Array{FileKey};
  set_names=[:training, :test],
  set_sizes=fill([0.7, 0.3], length(files)),
  select_channels=0:39
  )

  tierW_bindings = TTreeBindings()
  evtNo = tierW_bindings[Symbol("tier2.WaveletCoeffs_evtNo")] = zeros(Int32, 0)
  channels = tierW_bindings[Symbol("tier2.WaveletCoeffs_ch")] = zeros(Int32, 0)

  # std:vector containing wavelet coefficients
  # IMPORTANT: indexing on std:vector uses c-style integers 0 to len-1
  samples_cxx = tierW_bindings[Symbol("tier2.WaveletCoeffs_samples")] =
      ROOTFramework.CxxObjWithPtrRef(icxx"std::vector<std::vector<double>>();")

  # tier2.WaveletCoeffs_wlCoeffs, tier2.WaveletCoeffs_samples

  tier4_bindings = TTreeBindings()
  energies = tier4_bindings[:energy] = zeros(Float64, 0)
  event_ch = tier4_bindings[:eventChannelNumber] = Ref(zero(Int32))
  # timestamp = tier4_bindings[:timestamp] = Ref(zero(UInt64))
  aoeVeto = tier4_bindings[:psdFlag_AoE] = zeros(Int32, 0)
  aoeEval = tier4_bindings[:psdIsEval_AoE] = Bool[]
  aoeVal = tier4_bindings[:psdClassifier_AoE] = zeros(Float64, 0)
  samples = Float64[]

  # Prepare tables
  result = Dict{Symbol, EventLibrary}()
  waveform_lists::Dict{Symbol, Array{Array{Float32}}} = Dict()
  for (i,set_name) in enumerate(set_names)
    events = EventLibrary(zeros(Float32, 0, 0))
    events.labels[:E] = Float32[]
    events.labels[:AoE] = Float32[]
    events.labels[:AoE_class] = Float32[]
    events.prop[:name] = string(set_name)
    result[set_name] = events
    waveform_lists[set_name] = Array{Float32}[]
  end


  @time for (file_i,filekey) in enumerate(files)
    fileW = path(base_path, filekey, :tierW)
    file4 = path(base_path, filekey, :tier4)
    if !isfile(fileW) || !isfile(file4)
      println("Skipping because files don't exist: $fileW")
    else
      tierW_tchain = TChain("tree", fileW)
      tier4_tchain = TChain("tier4", file4)

      assert(length(tierW_tchain) == length(tier4_tchain))
      n = length(tierW_tchain)
      println("Reading file $file_i / $(length(files)). Entries: $n")

      tierW_input = TTreeInput(tierW_tchain, tierW_bindings)
      tier4_input = TTreeInput(tier4_tchain, tier4_bindings)

      switchdict = setsplit(set_names, set_sizes[file_i], n)

      tab_energies, tab_aoeValues, tab_aoeClasses, list_waveforms = lookup(result, waveform_lists, set_names[1])

      for i in zip(tierW_input, tier4_input)
          for detector_i in select_channels
                detector = detector_i + 1
            if length(samples_cxx.x[detector-1]) > 0
              resize!(samples, length(samples_cxx.x[detector-1]))
              copy!(samples, samples_cxx.x[detector-1])

              # Add new values to lists and arrays
              push!(tab_energies, energies[detector])
              push!(tab_aoeValues, aoeVal[detector])
              push!(list_waveforms, convert(Array{Float32}, samples))
              if aoeEval[detector]
                  push!(tab_aoeClasses, aoeVeto[detector])
              else
                  push!(tab_aoeClasses, -1)
              end
            end
          end

          # switch sets
          if haskey(switchdict, i[1])
            nextset = switchdict[i[1]]
            tab_energies, tab_aoeValues, tab_aoeClasses, list_waveforms = lookup(result, waveform_lists, nextset)
          end
        end
      end
    end

  for set_name in set_names
    if length(waveform_lists[set_name]) > 0
      tab_waveforms = hcat(waveform_lists[set_name]...)
    else
      tab_waveforms = zeros(Float32, 0,0)
    end
    result[set_name].waveforms = tab_waveforms
  end

  return result
end

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

function lookup(result, waveform_lists, set_name)
  tab_energies = result[set_name].labels[:E]
  tab_aoeValues = result[set_name].labels[:AoE]
  tab_aoeClasses = result[set_name].labels[:AoE_class]
  list_waveforms = waveform_lists[set_name]
  return tab_energies, tab_aoeValues, tab_aoeClasses, list_waveforms
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
#     println("Reading file $file")
#     chain1 = TChain("MGTree", file)
#     data = TTreeInput(chain1, tier1)
#     n = @cxx chain1->GetEntries()
#     println("Tree has length $n")
#
#     for i in data
#       println(i)
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
