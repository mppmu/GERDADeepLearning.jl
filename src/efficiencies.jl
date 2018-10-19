# This file is a part of GERDADeepLearning.jl, licensed under the MIT License (MIT).

using StatsBase, LsqFit


export EfficiencyCurve
type EfficiencyCurve
  name :: Symbol # Tl_DEP, Bi_FEP, Tl_SEP, Tl_FEP, :Phy2v
  cut_values :: Vector{Float64} # in ascending order
  efficiencies :: Vector{Float64}
  std :: Vector{Float64} # zeros if not used
  # count_before :: Float64
  # counts_after :: Vector{Float64}

  EfficiencyCurve(name::Symbol, cut_values::Vector{Float64}) = new(name, cut_values, zeros(Float64, length(cut_values)), zeros(Float64, length(cut_values)))
  EfficiencyCurve(name :: Symbol, cut_values :: Vector{Float64}, efficiencies :: Vector{Float64}, std :: Vector{Float64}) = new(name, cut_values, efficiencies, std)
end

export EfficiencyCollection
type EfficiencyCollection
  curves :: Array{EfficiencyCurve}
  name :: AbstractString
  title :: AbstractString
end


Base.getindex(curve::EfficiencyCurve, range) = EfficiencyCurve(curve.name, curve.cut_values[range], curve.efficiencies[range], curve.std[range])

function Base.getindex(curves::EfficiencyCollection, name::Symbol)
  indices = find(curve -> curve.name == name, curves.curves)
  if length(indices) != 1
    throw(ArgumentError("Efficiency curve not contained: $name"))
  end
  return curves.curves[indices[1]]
end

Base.getindex(curves::EfficiencyCollection, range) = EfficiencyCollection([curve[range] for curve in curves.curves], curves.name, curves.title)


export save_effs
function save_effs(env::DLEnv, effs::EfficiencyCollection)
  dir = resolvepath(env, "models", "efficiencies-$(effs.name)")
  isdir(dir) || mkdir(dir)
  for curve in effs.curves
    save_eff_txt(curve, joinpath(dir, "$(curve.name).txt"))
  end
end

export load_effs
function load_effs(env::DLEnv, name::AbstractString)
  dir = resolvepath(env, "models", "efficiencies-$(name)")
  curves = EfficiencyCurve[]
  for filename in readdir(dir)
    cname = filename
    if endswith(cname, ".txt")
      cname = cname[1:end-4]
    end
    curve = load_eff_txt(joinpath(dir, filename), Symbol(cname))
    push!(curves, curve)
  end
  return sort_peaks!(EfficiencyCollection(curves, name, name))
end

function sort_peaks!(curves::EfficiencyCollection)
  order = [:Tl_DEP, :Bi_FEP, :Tl_SEP, :Tl_FEP]
  list = EfficiencyCurve[]
  for key in order
    push!(list, curves[key])
  end
  curves.curves = list
  return curves
end

export save_eff_txt
function save_eff_txt(eff::EfficiencyCurve, filename::AbstractString)
  if eff.std != nothing
    writedlm(filename, hcat(eff.cut_values, eff.efficiencies, eff.std))
  else
    writedlm(filename, hcat(eff.cut_values, eff.efficiencies))
  end
end

export load_effs_txt
function load_effs_txt(filenames::Vector{String}, names::Vector{Symbol}, name::AbstractString)
  curves = [load_eff_txt(filenames[i], names[i]) for i in 1:length(filenames)]
  return EfficiencyCollection(curves, name, name)
end

export load_eff_txt
function load_eff_txt(filename::AbstractString, name::Symbol)
  data = readdlm(filename)
  cut_values = data[:,1]

  efficiencies = data[:,2]
  # Efficiencies may be in percent
  if maximum(efficiencies) > 10
    efficiencies /= 100
  end

  # Std if available
  if size(data, 2) > 2
    std = data[:,3]
  else
    std = zeros(Float64, length(cut_values))
  end
  # Order cut values so they are ascending
  if cut_values[1] > cut_values[end]
    cut_values = cut_values[end:-1:1]
    efficiencies = efficiencies[end:-1:1]
    std = std[end:-1:1]
  end
  return EfficiencyCurve(name, cut_values, efficiencies, std)
end


export random_guessing
function random_guessing()
  sig = EfficiencyCurve(:Tl_DEP, [0.0, 1], [0.0, 1], [0.0, 0])
  bkg = EfficiencyCurve(:Bi_FEP, [0.0, 1], [0.0, 1], [0.0, 0])
  return EfficiencyCollection([sig, bkg], "rand", "Random guessing")
end


export get_peak_efficiencies
function get_peak_efficiencies(env::DLEnv, lib::EventLibrary, name::AbstractString)
  return get_peak_efficiencies(env, lib, name, name)
end
function get_peak_efficiencies(env::DLEnv, lib::EventLibrary, name::AbstractString, title::AbstractString)
  if isdir(resolvepath(env, "models", "efficiencies-$(name)"))
    effs = load_effs(env, name)
  else
    effs = peak_efficiency_curves(lib, name)
    effs = cut_efficiency_fluctuations(effs)
    save_effs(env, effs)
  end
  effs.title = title
  return effs
end

export peak_efficiency_curves
function peak_efficiency_curves(lib::EventLibrary, name::AbstractString; cut_values=100, peak_names::Vector{Symbol}=[:Tl_DEP, :Bi_FEP, :Tl_SEP, :Tl_FEP], peak_centers::Vector{Float64}=[1592.5, 1620.7, 2103.5, 2614.5], half_window::Float64=10.0, bin_width::Float64=0.5, psd_key=:psd)
  if isa(cut_values, Integer)
    cut_values = equal_counts_cut_values(lib, cut_values; psd_key=psd_key)
  else
        cut_values = collect(cut_values)
  end
  curves = [peak_efficiency_curve(peak_names[i], lib, peak_centers[i], cut_values; half_window=half_window, bin_width=bin_width, psd_key=psd_key) for i in 1:length(peak_centers)]
  return EfficiencyCollection(curves, name, name)
end

export peak_efficiency_curve
"""
Cut values must be in ascending order.
"""
function peak_efficiency_curve(curve_name::Symbol, lib::EventLibrary, peak_center::Float64, cut_values::Array{Float64}; half_window::Float64=10.0, bin_width::Float64=0.5, psd_key=:psd)
  curve = EfficiencyCurve(curve_name, cut_values)

  start_energy = peak_center-half_window-bin_width
  end_energy = peak_center+half_window+bin_width
  nbins = Int64(round((end_energy-start_energy)/bin_width))

  bin_edges = linspace(start_energy, end_energy, nbins)
  bin_width = bin_edges[2] - bin_edges[1]

  lib = filter(lib, :E, E -> (E>start_energy) && (E<end_energy))

  energy_hist = fit(Histogram, lib[:E], bin_edges)
  amplitude, error = fit_peak(bin_edges, energy_hist.weights, peak_center, half_window)
  amplitude_before = max(0.1, amplitude)

  for (i,cut_value) in enumerate(cut_values)
    filter!(lib, psd_key, psd -> psd > cut_value)
    energy_hist = fit(Histogram, lib[:E], bin_edges)
    amplitude, error = fit_peak(bin_edges, energy_hist.weights, peak_center, half_window)
    amplitude = max(0, amplitude)
    curve.efficiencies[i] = amplitude / amplitude_before
    curve.std[i] = sqrt(amplitude * (1- min(1,amplitude/amplitude_before))) / amplitude_before
  end

  return curve
end


"""
Hitogram fit: Fit a gaussian with variable width + simple background model to every peak in peak_centers. The background is taken from the half_window.
Points outside half_window+margin are thrown away.

Returns the best fit amplitude with respective error of the peak height in absolut event counts.
"""
function fit_peak(total_x_axis, total_energy_hist::Array, peak_center, half_window)
  # Define peak function (Gaussian)
  # Parameters: Constant, Linear, Gauss amplitude, Gauss sigma
  peak(e, p) = p[1] + p[2]*(e-peak_center) + p[3] * exp(- (e-peak_center).^2 ./ (2*p[4]^2))

  indices = find(x -> (x >= peak_center-half_window) && (x <= peak_center+half_window), total_x_axis)
  x_axis = total_x_axis[indices]
  energy_hist = total_energy_hist[indices]
  energy_hist = convert(Array{Float64}, energy_hist)

  mean_count = mean(energy_hist)

  fit_result = curve_fit(peak, x_axis, energy_hist, [mean_count*0.8, 0.0, mean_count*0.2, 3.0])
  fit_amplitude = fit_result.param[3]
  try
    fit_error = estimate_errors(fit_result)
    fit_amplitude_error = fit_error[3]
    return fit_amplitude, fit_amplitude_error
  catch exc
    info("Fit Errors could not be calculated $exc")
    return fit_amplitude, Inf
  end
end


export equal_counts_cut_values
"""
Returns an array of count cut values in descending order the first of which is max_cut_value (all events excluded). The cut values are derived from the classifier distribution so that the total event count decreases with constant speed.
"""
function equal_counts_cut_values(events::EventLibrary, count; psd_key=:psd)
  psd = events[psd_key]
  bin_edges = linspace(Float64(minimum(psd)), maximum(psd), count*10)
  psd_hist = fit(Histogram, psd, bin_edges)
  psd_counts = psd_hist.weights

  target_sizes = linspace(0, eventcount(events), count)
  cut_values = [bin_edges[_integration_index(psd_counts, target_sizes[i])] for i in 1:length(target_sizes)]
  return cut_values
end
function _integration_index(hist, target)
  sum = 0
  for (i,val) in enumerate(hist)
    if sum >= target return i end
    sum += val
  end
  return length(hist)
end


export cut_efficiency_fluctuations
function cut_efficiency_fluctuations(effs::EfficiencyCollection)
  for i in 1:length(effs.curves[1].cut_values)
    for curve in effs.curves
      if curve.std[i] > curve.efficiencies[i]
        info("Cut efficiencies at index $i due to high statistical error.")
        return effs[1:i]
      end
    end
  end
  return effs
end


export background_rejection_at
function background_rejection_at(signal_eff::AbstractFloat, effs::EfficiencyCollection; signal_peak=:Tl_DEP, bkg_peak=:Bi_FEP)
  sig = effs[signal_peak]
  bkg = effs[bkg_peak]

  for i in length(sig.cut_values):-1:1
    if sig.efficiencies[i] > signal_eff
      return i, sig.cut_values[i], sig.efficiencies[i], 1 - bkg.efficiencies[i]
    end
  end
  info("Signal efficiency $signal_eff is never reached!")
end

export background_rejection_std_at
function background_rejection_std_at(signal_eff::AbstractFloat, effs::EfficiencyCollection; signal_peak=:Tl_DEP, bkg_peak=:Bi_FEP)
    sig = effs[signal_peak]
    bkg = effs[bkg_peak]
    i, cut, eff, rej = background_rejection_at(signal_eff, effs; signal_peak=signal_peak, bkg_peak=bkg_peak)
    if (sig.std[i] == 0) || (bkg.std[i] == 0)
        return -1
    else
        return sqrt(sig.std[i]^2 + bkg.std[i]^2)
    end
end


export equal_rejection_and_efficiency
function equal_rejection_and_efficiency(effs::EfficiencyCollection; signal_peak=:Tl_DEP, bkg_peak=:Bi_FEP)
  sig = effs[signal_peak]
  bkg = effs[bkg_peak]

  for i in length(sig.cut_values):-1:1
    if sig.efficiencies[i] > 1-bkg.efficiencies[i]
      return i, sig.cut_values[i], sig.efficiencies[i], 1 - bkg.efficiencies[i]
    end
  end
  info("Signal efficiency is never better than background rejection!")
end


export find_cut_value
function find_cut_value(effs::EfficiencyCollection, cut_value)
  if isa(cut_value, Real)
    return cut_value
  elseif cut_value == "90% DEP"
    cut_value_result = background_rejection_at(0.9, effs)[2]
  elseif cut_value == "Equal"
    cut_value_result = equal_rejection_and_efficiency(effs)[2]
  else
    info("Illegal cut_value argument: $cut_value")
    return cut_value
  end
  info("Determined cut value $cut_value = $cut_value_result")
  cut_value_result
end


export curve
curve(eff::EfficiencyCurve) = (eff.cut_values, eff.efficiencies)

export roc_curve
""" Complete: :left, :right, true, false
"""
function roc_curve(effs::EfficiencyCollection; signal_effs=linspace(0.01, 0.99, 400).^0.1, signal_peak=:Tl_DEP, bkg_peak=:Bi_FEP, complete=true)

  real_signal_effs = Float64[]
  real_bkg_rej = Float64[]

  if complete == true || complete == :left
    push!(real_signal_effs, 0)
    push!(real_bkg_rej, 1)
  end

  for signal_eff in signal_effs
    values = background_rejection_at(signal_eff, effs; signal_peak=signal_peak, bkg_peak=bkg_peak)
    if values != nothing
      push!(real_signal_effs, values[3])
      push!(real_bkg_rej, values[4])
    else break
    end
  end

  if complete == true || complete == :right
    push!(real_signal_effs, 1)
    push!(real_bkg_rej, 0)
  end

  return real_signal_effs, real_bkg_rej
end


export load_current_effs
function load_current_effs(detector_name::AbstractString)
  target_peaks = [:Tl_DEP, :Bi_FEP, :Tl_SEP, :Tl_FEP]
  current_eff = nothing
  try
    if !contains(detector_name, "GD") # coax
      detector_id = get_detector_index(detector_name)
      files = ["/home/iwsatlas1/pholl/workspace/Andreas Efficiencies/$det-$(lpad(detector_id,2,0))-$pname.txt" for pname in ["208Tl DEP", "212Bi FEP", "208Tl SEP", "208Tl FEP"]]
      return load_effs_txt(files, target_peaks, "Current implementation")
    else # BEGe
      files = ["/home/iwsatlas1/pholl/workspace/AoE Efficiencies/efficiency_$(pname)_$(detector_name).txt" for pname in ["DEP", "FEP", "SEP", "FEP2"]]
      return load_effs_txt(files, target_peaks, "A/E")
    end
  catch err
    info("Could not load current efficiencies for detector $detector_name.")
  end
end
