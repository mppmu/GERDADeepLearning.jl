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
"path": "/remote/ceph/group/gerda/data/phase2/blind/v02.06/gen",
"keylists": ["run0062.txt", "run0063.txt"],
"sets":{
	"train": [0.9, 0],
	"xval": [0.1, 0],
	"test": [0, 1]
},
"detectors": [22, 30],
"cache": true,
"preprocessing": ["normalize_energy"]
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
```julia
# Create an EventLibrary from a matrix of waveforms (Float32 or Float64)
events = EventLibrary(randn(Float32, 256, N))

# Access waveforms
events.waveforms # Matrix{Float32}

# Add an attribute per waveform
put_label!(events, :MyAttribute, randn(Float32, N))

# Other attributes
name(events) # retrieves the name as a String
setname!(events, "myevents")

N = length(events) # get the number of events

selection = events[1:100] # get first 100 events
selection = filter(events, :MyAttribute, x -> x > 0) # filter events by label
events2 = copy(events) # create a shallow copy (references same waveforms)

# IO functions are usually handled internally by DLEnv.

get_classifiers(events); push_classifier(events, "Autoencoder") # used as labels by plotting commands
```


## Reading GERDA data
Reading the raw data from tier1 and tier4 as defined in the configuration is very simple.
```julia
datasets = getdata(env) # Dict{Symbol,EventLibrary}
```
This command reads all files referenced by the keylists in the configuration file and returns a dictionary mapping dataset names to the corresponding `EventLibrary`. Individual sets can be accessed through their declared name as a `Symbol`.
```julia
training_set = datasets[:train]
```
If the `cache` in the configuration is set to true, this will cause either the data to be read from cache if available or a cache to be created.

## Data processing: Defining a processing chain
Many processing steps take a `Dict{Symbol,EventLibrary}` as input and return one. As processing might be computationally expensive, it is usually useful to cache the result of a step and skip processing next time.

The GERDADeepLearning framework provides a simple method to handle process chains comfortably.
Each processing step is passed to the environment through the `get` method. Additionally the name of the result and a list of future steps that depend on it are passed.
```julia
preprocessed = get(env, "preprocessed"; targets=["latent", "dnn"]) do
  p=preprocess(env, sets)
  plot_waveforms(env, p[:test])
  return p
end
```
This step will only be executed and cached if no cache from a previous run exists. This feature can be disabled with the `cache` property in the configuration file.

## Machine learning algorithms
Currently the framework supports convolutional autoencoders and DNN classifiers.

```julia
#  Train and evaluate an autoencoder
lat = get(env, "latent") do
  ae = autoencoder(env, pre[:train], pre[:xval])
  plot_autoencoder(env, ae, pre[:xval])
  return encode(env, pre, ae)
end

# Train and evaluate a DNN
dnn_classifier(env, lat; id="latent-dnn-classifier", evaluate=[:train, :test])

```

```json

```

## Plotting

```julia
plot_autoencoder(env, ae, pre[:xval])
plot_classifier(env, "comparison", lat[:test], lat[:train], pre[:test], pre[:train])
```
