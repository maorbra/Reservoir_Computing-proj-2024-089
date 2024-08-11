# Reservoir Computing
This project is a collection of scripts for experimental implementation of a reservoir computer in optoelectronic delay system.

## Prerequisites
The code in this project is designed to control the following instruments
- **Agilent 3220A** arbitrary waveform generator
- **Liquid Instruments Moku:Go** (programmable FPGA delay, data logger, power supply)
- **PicoScope 5000 series** oscilloscope



## Installation
### Keysight I/O Drivers
Install [Keysight VISA](https://www.keysight.com/find/iosuiteproductcounter) to communicate with Agilent waveform generator.

### Python environment
Install project's python dependencies using [pipenv](https://pipenv.pypa.io/en/latest/installation.html)
```commandline
pipenv install
```
After installing pipenv environment, activate it in terminal by executing `pipenv shell` or configure your IDE to use it.

## Usage
### Prepare Datasets
Generate **Sin-Square** dataset
```commandline
python sin_square.py
```
Generate **NARMA10** dataset
```commandline
python narma10.py
```

Generate **Japanese Vowels** dataset
```commandline
python japanese_vowels.py
```

### Train Reservoir on a Dataset
Run reservoir training on a pickled dataset, for instance
```commandline
python train.py data/sin_square.pickle --test-size 0.5 --ridge 1e-4 --simulation
```
Note that by adding `--simulation` argument simulation is invoked.

Various parameters of the reservoir can be controlled using command line arguments, run help to see the list of available parameters:
```commandline
python train.py --help
```

### Calibrate Electro-Optical Devices
To recalibrate Mach-Zehnder modulator and laser power voltage control, run
```commandline
python calibrate.py
```
To plot the visualize the calibration  run
```commandline
python show_calibration.py
```

### Run Optimization
Run reservoir's hyperparameters optimization
```commandline
python rc_hyperopt.py
```

The script **rc_hyperopt.py** loads hyperparameters search space and other configuration parameters from **search_space.json** file.
You can create your own configuration files and call the script passing path to config file using ``--config`` argument
```commandline
python rc_hyperopt.py --config path_to_your_config.json
```

### Analyze Optimization Results
```commandline
python analyze_stats.py hyperopt_sin_square_simulation
```