# GERDADeepLearning


GERDA deep learning is an all-in-one solution for processing GERDA calibration and physics data using various neural network architectures.

Features
- Loading of ROOT files (tier1, tier4) from keylists
- Signal processing
- Training of convolutional autoencoders and DNNs
- Various plotting convenience functions
- Automatic caching and checkpoint creation
- Cohesive and easy to use API



## Defining an environment
GERDADeepLearning defines the type `DLEnv` which fixes the directory structure of the project and contains the configuration.
The project configuration is read from a JSON file upon creation of a `DLEnv`.

```json
# config.json example
{
	"path": "/remote/ceph/group/gerda/data/psa-skim/pholl/ch09",
	"keylists": ["run0063-cal-some.txt", "run0062-cal-some.txt"],
	"sets":{
		"train": [0.9, 0],
		"xval": [0.1, 0],
		"test": [0, 1]
	},
	"detectors": [9],
	"test-pulses": "exclude",
	"baseline-events": "include",
	"unphysical-events": "only",

	"preprocessing": ["baseline", "normalize_energy", "align_peaks"],

	"cache": true
}
```
The configuration file references all keylists which belong to the project and how to split their events into custom data sets. The example above creates a training set from 90% of the events from run 62, a cross validation set from the remaining 10% and a test set from run 63.

To create an environment, type on of the following.
```julia
using GERDADeepLearning
env = DLEnv() # reads "config.json" in the current directory
env = DLEnv("myconfig.json") # use a custom JSON in the current directory
env = DLEnv("/mypath", "myconfig.json") # custom path and config file
```

## Events and waveforms
Lists of events with corresponding waveforms and other properties are encapsulated in the `EventLibrary` type.
Waveforms are always stored column-wise in a matrix of `Float32`.

The following code shows how to create an EventLibrary and access the contained data. Once initialized, the waveforms of an EventLibrary should not be edited anymore.
```julia
# Create an EventLibrary from a matrix of waveforms (Float32 or Float64)
events = EventLibrary(randn(Float32, 256, N))

# Access waveforms as Matrix{Float32}
wf = waveforms(events)

# get the number of events
N = length(events)
```

In addition to the waveform, events can be assigned further attributes. These are stored as a `Vector` of type `Float32`. The length of the vector must be equal to the number of events in the EventLibrary.
```julia
# Add an attribute of type Float32 to each event
put_label!(events, :MyAttribute, randn(Float32, N))

# Access the attribute
myattr = events[:MyAttribute]
```

Each EventLibrary has a name which is used e.g. for naming files generated from these events. All libraries created from methods in GERDADeepLearning are given default names. If no name is set, the EventLibrary will show up as `<unnamed>`.
```julia
evtname = name(events) # String
setname!(events, "myevents")
```

Other properties that are commonly added to EventLibraries are `:preprocessing`, `waveform_type`.

It is often useful to access only a part of all events contained in an EventLibrary. EventLibraries therefore support subsets through indexing as well as filtering events by a specific attribute.
```julia
# get first 100 events
selection = events[1:100]

# filter events by label
selection = filter(events, :MyAttribute, x -> x > 0)
```

The waveform array of an EventLibrary should not be edited after the library is created. To derive new libraries from old ones, it is often useful to copy these first. Copy methods not only copy the waveforms but also event attributes and other properties of the library.
```julia
# create a shallow copy, references same waveforms and label arrays
copy(events)

# Also create copies of waveforms and label arrays
deepcopy(events)
```

GERDADeepLearning provides convenience functions for plotting waveforms.
```julia
# Plot a number of waveforms in one diagram
plot_waveforms(env::DLEnv, events::EventLibrary; count=4, bin_width_ns=10, cut=nothing)
plot_waveforms(data::Array{Float32, 2}, filepath::AbstractString;
      bin_width_ns=10, cut=nothing, diagram_font=font(16))
```

Reading and writing EventLibraries is usually handled by the framework and not the developer explicitely. Using the `get` function ensures that all progress of your script is saved automatically.


## Reading & Preprocessing GERDA Data
Reading the raw data from tier1 and tier4 as defined in the configuration is very simple.
```julia
datasets = getdata(env; preprocessed=false) # Dict{Symbol,EventLibrary}
```
This command reads all files referenced by the keylists in the configuration file and returns a dictionary mapping dataset names to the corresponding `EventLibrary`. Individual sets can be accessed through their declared name as a `Symbol`.
```julia
training_set = datasets[:train]
```
If the argument `preprocessed` is set to `true`, the preprocessing chain defined in the config file will be executed and the preprocessed data is returned.

Supported preprocessing functions are
```
baseline # Subtract starting baseline and invert the charge pulse.
normalize_energy # Normalize end of charge pulse to one.
align_midpoints # Aligns the 50% positions to the center cutting a 256 samples window.
align_peaks # Aligns the maximum values in the current pulse to the center cutting a 256 samples window.

# Switch between charge / current representations
integrate
differentiate

HE # Select only pulses with E > 1500 keV.
```

If the `cache` in the configuration is set to true, this will cause either the data to be read from cache if available or a cache to be created. Note that the raw data and the preprocessed data are cached in different files.

Note that the framework checks whether the the cache of `preprocessed` is up to date but does not check whether the data cache is still valid. Therefore either delete the data data.h5 directly or through `delete!(env,"data")` after changing your data references in the configuration.

## Data processing: Defining a processing chain
Many processing steps take a `Dict{Symbol,EventLibrary}` as input and return one. As processing might be computationally expensive, it is usually useful to cache the result of a step and skip processing next time.

The GERDADeepLearning framework provides a simple method to handle process chains comfortably.
Each processing step is passed to the environment through the `get` method. Additionally the name of the result and a list of future steps that depend on it are passed.
```julia
preprocessed = get(env, "preprocessed"; targets=["latent", "dnn"]) do
  p=preprocess(env, sets)
  plot_waveforms(env, p) # returns p
end
```
This step will only be executed and cached if no cache from a previous run exists. This feature can be disabled with the `cache` property in the configuration file.

## Machine learning algorithms
Currently the framework supports convolutional autoencoders and DNN classifiers.
Every instance of neural network has a unique ID of type `String`. The network layout and learning parameters are stored within the JSON configuration file in a subset with the corresponding ID.

For two networks with IDs `autoencoder` and `dnn-classifier` the configuration file would have the following structure.
```json
{
"path": "/remote/ceph/group/gerda/data/psa-skim/pholl/ch09",
"keylists": ["run0063-cal-allFiles.txt", "run0062-cal-allFiles.txt"],

"autoencoder":
{   },
"dnn-classifier":
{   }
}
```

Custom configurations can also be generated at runtime from existing templates.
```julia
# Create a modified autoencoder specification with a different batch size
new_properties!(env, "autoencoder", "autoencoder2") do p
	p["batch_size"] = 200
end
```

Networks can be obtained through the `autoencoder` and `dnn_classifier` methods which return a network handle. How this network is obtained depends on the `action` parameter.
Supported actions are:
- `:train` to train a new network from scratch (replaces existing network with same ID) and save it in the environment.
- `:load` to load a previously trained network.
- `:refine` to load a previously trained network and continue training until the specified number of epochs is reached.
- `:auto` (default) to let the library decide what action to take based upon previously saved data.

In any case, the returned network handle is initialized with the parameters that gave the best result on the validation set.

### Autoencoder
Use a convolutional autoencoder to move back and forth between original representation and compact representation (latent space).

```julia
# data::Dict{Symbol, EventLibrary}
# dataset::EventLibrary

# Obtain an autoencoder (load or train)
net = autoencoder(env, data; id="autoencoder", action=:auto, train_key=:train, xval_key=:xval)

compact = encode(data, net)
compact = encode(dataset, net)

reconst = decode(dataset, net, target_size)

reconst = encode_decode(dataset, net)

error = mse(net, dataset) # Returns the mean squared error averaged over all events

# Train a custom decoder
dec = decoder(env, latent_data, target_data; id="decoder", action::Symbol=:auto, train_key=:train, xval_key=:xval)
encode_decode(latent_dataset, dec)
```


Supported parameters in configuration file:
```json
"autoencoder":
{
	"slim": 0,

	"conv_filters": [4, 8],
	"conv_lengths": [9, 9],
	"pool_size": [4, 4],
	"pool_type": "max",
	"conv_dropout": 0.0,

	"fc": [10],
	"dropout": 0.0,
	"activation": "relu",

	"learning_rate": 0.001,
	"batch_size": 100,
	"epochs": 30
}
```

Visualizing an autoencoder can simply be done using one of the `plot_autoencoder` functions. These will plot the training curves, the convolutional filters and, when provided with events, a number of reconstructions.
```julia
# Plots reconstructions from the validation set and network visualization
plot_autoencoder(env::DLEnv, n::NetworkInfo) # only network
plot_autoencoder(env::DLEnv, n::NetworkInfo, data::Dict{Symbol,EventLibrary}; count=20, transform=identity) # network and reconstruction
```



### DNN classifier
Use a neural network with fully connected layers to classify a dataset.

```julia
# Using a DNN for classification
dnn = dnn_classifier(env, data; id="latent-dnn-classifier", action=:auto, label_key=:SSE, train_key=:train, xval_key=:xval, evaluate=[:train, :test])

predict(dataset, net; psd_name=:psd) # Adds a label to the EventLibrary
```

Supported parameters in configuration file:
```json
"dnn-classifier":
{
	"slim": 0,

	"fc": [20, 10],
	"dropout": 0.5,
	"activation": "relu",

	"learning_rate": 0.001,
	"batch_size": 100,
	"epochs": 1000
}
```

Evaluating and plotting the performance of a trained classifier can be done simply by calling `plot_classifier`. This method takes any number of classified and labeled EventLibraries, calculates the efficiencies and plots various distributions and curves.
```julia
# Evaluates classifier efficiencies on the given datasets and plots results
plot_classifier(env::DLEnv, name, libs::EventLibrary...;
    classifier_key=:psd, label_key=:SSE, plot_AoE=false)
```
