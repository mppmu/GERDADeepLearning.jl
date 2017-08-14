# This file is a part of GERDADeepLearning.jl, licensed under the MIT License (MIT).

using MXNet, Plots, StatsBase

pyplot() # workaround for Julia 0.6 Plots.jl bug

function plot_learning_curves(n::NetworkInfo, filename; from_zero::Bool=false)
  train = n.training_curve
  xval = n.xval_curve
  plot(size=(800,500), legendfont=font(20))
  plot!(train, label="Training set", line=2)
  plot!(xval, label="Validation set", line=2)
  xaxis!("Training time (epochs)", font(20))
  yaxis!("Mean squared error", font(20))
  if from_zero
    yaxis!((0, max(maximum(train), maximum(xval))))
  end
  savefig("$filename.pdf")
  savefig("$filename.png")
end

export plot_dnn
function plot_dnn(env::DLEnv, n::NetworkInfo)
  dir = joinpath(env.dir, "plots", n.name)
  isdir(dir) || mkdir(dir)
  info(env, 2, "Generating plots in $dir...")
  plot_learning_curves(n, joinpath(dir, "learning_curves"))
end

export plot_autoencoder
function plot_autoencoder(env::DLEnv, n::NetworkInfo)
  dir = joinpath(env.dir, "plots", n.name)
  isdir(dir) || mkdir(dir)
  info(env, 2, "Generating plots in $dir...")

  model = n.model

  plot_learning_curves(n, joinpath(dir, "learning_curves"); from_zero=true)
  visualize_1D_convolution(model, :conv_1_weight, joinpath(dir,"filters1.png"))
  visualize_2D_convolution(model, :conv_2_weight, joinpath(dir,"filter2"))
  info(env, 2, "Saved network plots to $dir")
  return model, dir
end

function plot_autoencoder(env::DLEnv, n::NetworkInfo, data::Dict{Symbol,EventLibrary}; count=20, transform=identity)
  for (key,value) in data
    plot_autoencoder(env, n, value; count=count, transform=transform)
  end
end

function plot_autoencoder(env::DLEnv, n::NetworkInfo, data::EventCollection; count=20, transform=identity)
  batch_size = n["batch_size"]

  model, dir = plot_autoencoder(env, n)

  data = flatten(data)

  if eventcount(data) > count
    data = data[1:count]
  end

  provider = padded_array_provider(:data, waveforms(data), batch_size)
  plot_reconstructions([n.model], ["Reconstruction"], waveforms(data), provider, dir, file_prefix=name(data)*"-", sample_count=eventcount(data), transform=transform, bin_width_ns=sampling_period(data), legendpos=(data[:waveform_type] == "current" ? :topright : :topleft))

  # loss = model.arch
  # open(joinpath(dir,"graphviz.dot"), "w") do file
  #   info(file, mx.to_graphviz(loss))
  # end

  info(env, 2, "Saved reconstruction plots to $dir")

  return model, dir
end


function plot_reconstructions(models, names, waveforms, provider, plot_dir;
    file_prefix="", sample_count=20, transform=identity,
    bin_width_ns=10, y_label="Current pulse", diagram_font=font(14), legendpos=:topright
    )
  # plot_waveform_comparisons(encode_decode(...), ...)
  pred_evals = [transform(mx.predict(model, provider)) for model in models]
  truth = transform(waveforms)

  sample_count = min(sample_count, size(truth, 2))
  x_axis = linspace(0, (size(truth,1)-1)*bin_width_ns*1e9, size(truth,1))

  reconst_animation = @animate for i in 1:sample_count
    plot(x_axis, truth[:,i], label="Data", linewidth=2, legendfont=diagram_font, legend=legendpos)
    for modelI in 1:length(models)
      plot!(x_axis, pred_evals[modelI][:,i], linewidth=3, label=names[modelI])
    end
   xaxis!("Time (ns)", diagram_font)
   yaxis!(y_label, diagram_font)
   savefig("$plot_dir/reconst-$(file_prefix)$i.png")
  end
  gif(reconst_animation, "$plot_dir/reconstructions.gif", fps=1)
end

export plot_waveform_comparisons
function plot_waveform_comparisons(env::DLEnv, libs::EventLibrary...; count=20, cut=nothing, diagram_font=font(14), transform=identity, title::String="", y_label="Current pulse")
  diag_count = min(count, minimum(length.(libs)))
  if diag_count != count
    info(env, 2, "Diagram count reduced to $diag_count, requested: $count")
  end
  sample_count = minimum(samples.(libs))
  x_axis = linspace(0, (sample_count-1)*sampling_period(libs[1])*1e9, sample_count)

  # Create directory
  dir = joinpath(env, "plots", title)
  isdir(dir) || mkdir(dir)
  info(env, 2, "Saving comparison plots to $dir ($(length(libs)) libraries)")

  for i in 1:diag_count
    plot(legendfont=diagram_font, legend=:topright)
    for lib in libs
      plot!(x_axis, waveforms(lib)[:,i], linewidth=2, label=name(lib))
    end
   xaxis!("Time (ns)", diagram_font)
   yaxis!(y_label, diagram_font)

   savefig(joinpath(dir, "wfcmp-$(join(name.(libs),"-"))$i.png"))
  end
  return libs
end


export plot_waveforms
"""
Plots a number of waveforms from the given EventLibrary in a single diagram. The figure is saved in the given environment using the name of the library.
"""
function plot_waveforms(env::DLEnv, events::EventLibrary; count=4, bin_width_ns=10, cut=nothing, diagram_font=font(16), evt_indices=nothing)
  filepath = joinpath(env.dir, "plots", "waveforms-$(name(events)).png")
  info(env, 2, "Plotting waveforms to $filepath")

  if evt_indices == nothing
    count = min(count, eventcount(events))
    data = events.waveforms[:,1:count]
  else
    data = events.waveforms[:, evt_indices]
  end

  if cut != nothing
    data = data[cut,:]
  end

  x_axis = linspace(0, (size(data,1)-1)*bin_width_ns, size(data,1))
  plot(size=(600, 400), legend=:none)
  for i in 1:size(data, 2)
    plot!(x_axis, data[:,i], linewidth=3)
  end
  xaxis!("Time (ns)", diagram_font)
  yaxis!("Current pulse", diagram_font)
  savefig(filepath)

  return events
end

"""
For each EventLibrary, a number of waveforms are plotted in one diagram. All figures are saved in the given environment using the names of the libraries.
"""
function plot_waveforms(env::DLEnv, data::DLData; count=4, bin_width_ns=10, cut=nothing)
  for lib in data
    plot_waveforms(env, lib; count=count, bin_width_ns=bin_width_ns, cut=cut)
  end
  return data
end

export plot_waveform_thumbnail
function plot_waveform_thumbnail(env::DLEnv, waveform::Vector{Float32}; bin_width_ns=10, tcut=50:200, titlestring=nothing, diagram_size=(140, 100), diagram_font=font(12), filename="waveform-thumbnail.png")
  waveform = waveform[tcut]
  tlength = tcut[end] - tcut[1]
  time = linspace(0, (tlength-1)*0.01, length(waveform))

  fig = plot(time, waveform, size=diagram_size, line=(:blue), legend=:none, xformatter = x -> "$(Int(x)) us", grid=false)
  yticks!(Float64[])
  xticks!([0,1])
  xaxis!(diagram_font)
  yaxis!((minimum(waveform), maximum(waveform)*1.1))
  if titlestring != nothing
    title!(titlestring, titlefont=diagram_font)
  end
  savefig(joinpath(env.dir, "plots", filename))
  fig
end


function visualize_1D_convolution(model, name, filename)
  for param in model.arg_params
    if(param[1] == name)
      filters = copy(param[2])[1,:,1,:] # One column per filter
      filters = vcat(filters, transpose(filters[end,:]))
      plot(filters, line=(2, :steppost))
      title!("Learned convolutional filters in first layer")
      xaxis!("Delta time")
      yaxis!("Conv. filter value")
      savefig(filename)
    end
  end
end

function visualize_2D_convolution(model, name, filename_base)
  for param in model.arg_params
    if(param[1] == name)
      for i in 1:size(param[2], 4)
        filters = copy(param[2])[1,:,:,i]
        heatmap(filters)
        title!("Learned convolutional filters in second layer")
        xaxis!("Delta time")
        yaxis!("Feature map")
        savefig("$filename_base-$i.png")
      end
    end
  end
end


export plot_classifier_histogram
function plot_classifier_histogram(dir, events::EventLibrary, label_key, psd_key;
    diagram_font=font(14), nbins = 30)

  if !haskey(events, psd_key)
    throw(ArgumentError("events must have label 'psd'."))
  end

  events = filter(events, :E, e -> (e < 3800 && e > 600))

  bins = linspace(minimum(events.labels[psd_key]), maximum(events.labels[psd_key]), nbins)

    # Labeled distribution
  if haskey(events, label_key)
    labels = events[label_key]
    psd = events[psd_key]

    psd_SSE = psd[find(l -> l==1, labels)]
    psd_MSE = psd[find(l -> l==0, labels)]

    # Overall PSD distribution
    histogram(psd_SSE, bins=bins, label="Tl DEP ($(length(psd_SSE)) events)", legendfont=diagram_font, linewidth=0)
    histogram!(psd_MSE, bins=bins, label="Bi FEP ($(length(psd_MSE)) events)", fillalpha=0.7, legendfont=diagram_font, linewidth=0)
    xaxis!("Classifier response", diagram_font)
    yaxis!("Event count", diagram_font)
    savefig(joinpath(dir,"Class distribution $(name(events)).png"))
  end

    # Total distribution
  histogram(events[psd_key], bins=bins, label="All ($(eventcount(events)) events)", legendfont=diagram_font, linewidth=0)
  xaxis!("Classifier response", diagram_font)
  yaxis!("Event count", diagram_font)
  savefig(joinpath(dir,"Distribution $(name(events)).png"))

    # Energy dependence
  e_dep_hist = fit(Histogram{Float64}, (convert(Array{Float64},events[:E]),
    convert(Array{Float64},events[psd_key])), (linspace(1000, 3000, 101), linspace(0, 1, 51)))
  broadcast!(x -> x <= 0 ? NaN : log10(x), e_dep_hist.weights, e_dep_hist.weights)
  plot(e_dep_hist, color=:viridis)
  xaxis!("Energy (keV)")
  yaxis!("Classifier output")
  title!("Classification vs energy (log10)")
  savefig(joinpath(dir, "Energy distributions $(name(events)).png"))

    # Time dependence
    events_sorted = sort(events, :timestamp)
  histogram2d(convert(Array{Float64}, 1:length(events_sorted[psd_key]))/1000, convert(Array{Float64},events_sorted[psd_key]))
    xaxis!("Entries (1000)", font(16))
    yaxis!("Classification", font(16))
    title!("Time stability of $(events[:detector_name])", titlefont=font(16))
  savefig(joinpath(dir, "Distribution over time $(name(events)) entries.png"))
  histogram2d(convert(Array{Float64}, (events_sorted[:timestamp]-events_sorted[:timestamp][1])/60/60/24), convert(Array{Float64},events_sorted[psd_key]))
    xaxis!("Time (days)", font(16))
    yaxis!("Classification", font(16))
    title!("Time stability of $(events[:detector_name])", titlefont=font(16))
  savefig(joinpath(dir, "Distribution over time $(name(events)) millis.png"))

  if haskey(events.labels, :multiplicity)
    histogram2d(convert(Array{Float64},events.labels[:multiplicity]), convert(Array{Float64},events.labels[psd_key]))
    savefig(joinpath(dir, "Multiplicity correlation $(name(events)).png"))
  end

  return bins
end

export plot_energy_histogram
function plot_energy_histogram(env::DLEnv, events::EventLibrary, psd_cut, psd_key)
  all = copy(events)
  all.prop[:diagram_label] = "All events"
  cut = filter(all, psd_key, psd -> psd > psd_cut)
  cut.prop[:diagram_label] = "After PSD, cut value $(@sprintf("%.2f",psd_cut))"
  # range = linspace(1500, 2800, 150)
  plot_energy_histogram([all, cut], 150,
      joinpath(env.dir, "plots", "PSD-energies-cut$(@sprintf("%.3f",psd_cut)).png"))
end

function plot_energy_histogram(eventlibs::Array{EventLibrary}, nbins, filename;
  diagram_font=font(14), label=:diagram_label)

  histogram(xticks=[2000, 2500], legend=:bottomleft, legendfont=diagram_font)

  # Peaks before and after
  for events in eventlibs
    histogram!(events[:E], nbins=nbins, label=events[label], linewidth=0)
  end
  xaxis!("Energy (keV)", diagram_font)
  yaxis!("Events", :log10, (1,Inf), diagram_font)
  savefig(filename)
end


export plot_efficiency_curves
function plot_efficiency_curves(file::AbstractString, effs::EfficiencyCollection)
  cut_values = effs.curves[1].cut_values

  plot(size=(600,400), legendfont=font(16))
  for curve in effs.curves
    # plot!(curve.cut_values, curve.efficiencies, marker=:circle, markerstrokewidth=0.2, label=replace(string(curve.name), "_", " "))
    plot!(curve.cut_values, curve.efficiencies, line=(3), label=replace(string(curve.name), "_", " "))
  end

  xaxis!("Cut value", font(20))
  yaxis!("Efficiency", (0,1), font(20))
  savefig("$file.png")
  savefig("$file.pdf")
  return effs
end


export plot_efficiencies
function plot_efficiencies(file::AbstractString, effs::EfficiencyCollection...)
  plot(size=(700, 600), legendfont=font(16), legend=:bottomleft)

  for eff in effs
    if length(eff.curves[1].cut_values) <= 2
      plot!(roc_curve(eff), label=eff.title, line=(:dash, :grey, 2))
    else
      plot!(roc_curve(eff), label=eff.title, linewidth=4)
    end
  end
  xaxis!("Signal efficiency", (0.6,1), font(20))
  yaxis!("Background rejection", (0,1), font(20))
  savefig("$file.png")
  savefig("$file.pdf")
  return effs
end


export plot_classifier
function plot_classifier(env::DLEnv, dirname::AbstractString, libs::EventLibrary...;
    classifier_key=:psd, label_key=:SSE, plot_AoE=false)
  dir = joinpath(env.dir, "plots", dirname)
  isdir(dir) || mkdir(dir)

  for lib in libs
    plot_classifier_histogram(dir, lib, label_key, classifier_key)
  end

  peak_effs = [get_peak_efficiencies(env, lib, lib[:detector_name], join(get_classifiers(lib), " + ")) for lib in libs]
  current_curve = load_current_effs(libs[1][:detector_name])

  for (i,lib) in enumerate(libs)
    plot_efficiency_curves(joinpath(dir, "PSD Efficiencies vs Cut $(name(lib))"), peak_effs[i])
  end
  if current_curve != nothing
    plot_efficiency_curves(joinpath(dir, "PSD Efficiencies vs Cut current"), current_curve)
  end

  curves = EfficiencyCollection[]
  if current_curve != nothing
    push!(curves, current_curve)
  end
  append!(curves, peak_effs)
  push!(curves, random_guessing())

  plot_efficiencies(joinpath(dir, "PSD Efficiency comparison"), curves...)
end
