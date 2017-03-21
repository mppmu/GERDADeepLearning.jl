# This file is a part of GERDADeepLearning.jl, licensed under the MIT License (MIT).

using StatsBase, LsqFit


function fit_peaks(energies, peak_centers; half_window=10.0, bin_width=0.5, margin=1.0)
  start_energy = peak_centers[1]-half_window-margin
  end_energy = peak_centers[end]+half_window+margin
  nbins = convert(Int64, round((end_energy-start_energy)/bin_width))

  bin_edges = linspace(start_energy, end_energy, nbins)
  bin_width = bin_edges[2] - bin_edges[1]
  energy_hist = fit(Histogram, energies, bin_edges)

  amplitudes = zeros(Float64, length(peak_centers))
  errors = zeros(Float64, length(peak_centers))

  for i in 1:length(peak_centers)
    amplitudes[i], errors[i] = fit_peak(bin_edges, energy_hist.weights, peak_centers[i], half_window)
  end
  return amplitudes, errors
end


function fit_peak(total_x_axis, total_energy_hist::Array, peak_center, half_window)
  # Define peak function (Gaussian)
  # Parameters: Constant, Linear, Gauss amplitude, Gauss sigma
  peak(e, p) = p[1] + p[2]*(e-peak_center) + p[3] * exp(- (e-peak_center).^2 ./ (2*p[4]^2))

  indices = find(x -> (x >= peak_center-half_window) && (x <= peak_center+half_window), total_x_axis)
  x_axis = total_x_axis[indices]
  energy_hist = total_energy_hist[indices]

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
function equal_counts_cut_values(events::EventLibrary, count, psd_key=:psd)
  psd = events[psd_key]
  bin_edges = linspace(minimum(psd), maximum(psd), count*5)
  psd_hist = fit(Histogram, psd, bin_edges)
  psd_counts = psd_hist.weights

  target_sizes = linspace(length(events), 0, count-1)
  cut_values = [bin_edges[_integration_index(psd_counts, target_sizes[i])] for i in 1:length(target_sizes)]
  return vcat(1.01, cut_values)
end

function _integration_index(hist, target)
  sum = 0
  for (i,val) in enumerate(hist)
    if sum >= target return i end
    sum += val
  end
  return length(hist)
end


export calc_efficiency_curves
function calc_efficiency_curves(events::EventLibrary, cut_key, cut_values,
      peak_centers;
      fit_half_window=10, fit_bin_width=0.5
  )
  println("Calculating efficiencies for $(length(cut_values)) cut values on set '$(name(events))'")

  filter(events, :E, e -> _keep(e, peak_centers, fit_half_window*1.5))

  amplitudes_before,amplitudes_before_errors = fit_peaks(events[:E], peak_centers)

  amplitudes_after = zeros(length(peak_centers), length(cut_values))
  amplitudes_after_errors = zeros(length(peak_centers), length(cut_values))

  for (i, cut) in enumerate(cut_values)
    cut_events = filter(events, cut_key, psd -> psd >= cut)
    # println("For cut $cut accept: $(length(cut_events)) / $(length(events))")
    amplitudes_after[:,i],amplitudes_after_errors[:,i] = fit_peaks(cut_events[:E], peak_centers)
  end

  # fit_file = open(joinpath(dir,"peak_fit.txt"), "w")
  # write(fit_file, "Peak energies, Amplitudes before, Amplitudes after, Ratios\n")
  # writedlm(fit_file, hcat(peak_centers, amplitudes_before, amplitudes_after, amplitudes_before./amplitudes_after))
  # close(fit_file)

  amplitudes_before = max.(0.1, amplitudes_before)
  amplitude_ratios = zeros(length(peak_centers), length(cut_values))
  for i in 1:length(cut_values)
    amplitudes_after[:,i] = max.(0, min.(amplitudes_before, amplitudes_after[:,i]))
    amplitude_ratios[:,i] = amplitudes_after[:,i] ./ amplitudes_before
  end
  # Binomial error calculation
  amplitude_errors = sqrt.(amplitudes_after .* (1- min(1,amplitudes_after./amplitudes_before))) ./ amplitudes_before

  return Dict{Symbol, Array{Float64, 2}}(
      :cut_value => transpose(cut_values),
      :before => hcat(fill(amplitudes_before, length(cut_values))...),
      :after => amplitudes_after,
      :ratio => amplitude_ratios,
      :ratio_err => amplitude_errors
  )
end

function _keep(e, peak_centers, margin)
  for peak_center in peak_centers
    if abs(e-peak_center) <= (margin)
      return true
    end
  end
  return false
end

function cut_efficiency_fluctuations(eff::Dict{Symbol, Array{Float64, 2}};
    max_cut_efficiency=0.2)
  ratios = eff[:ratio]
  peaks = 1:size(ratios,1)
  # ratio should never decrease
  for i in (size(ratios, 2)-1):-1:1
    for peak in peaks
      if (ratios[peak, i+1] < max_cut_efficiency) && (ratios[peak, i] > ratios[peak, i+1])
        return _arraysfrom(eff, i+1)
      end
    end
  end

  return eff
end

function _arraysfrom(eff::Dict{Symbol, Array{Float64, 2}}, startindex)
  result = Dict{Symbol, Array{Float64, 2}}()
  for (key, array) in eff
    result[key] = array[:,startindex:end]
  end
  return result
end
