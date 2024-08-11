import os.path
from collections import defaultdict
import subprocess
import time

import h5py
import pyvisa
from moku.instruments import FIRFilterBox
import numpy as np
from picoscope.ps5000a import PS5000a
from pymeasure.instruments.agilent import Agilent33220A
from pymeasure.instruments.keysight import KeysightDSOX1102G
import reservoirpy.mat_gen
from reservoirpy.nodes import Ridge
import scipy.linalg
import scipy.signal
import scipy.optimize
import scipy.interpolate
from tqdm import tqdm

from calc_vpi import retrieve_mzm_params
from mzm_map import mzm_func
from config import DATA_DIR
from instruments import raise_instrument_errors
from preprocess import join_items


WFG_ARB_MAX_POINTS = 65536
WFG_ARB_MAX_POINTS_FAST = 16384
WFG_DAC_MAX_INT = 8191


class ExperimentalReservoir:
    def __init__(
            self,
            optical_power,
            phi0,
            nodes,
            seed,
            neuron_duration,
            logging_mode,
            logging_rate,
            fir_in_gain,
            fir_rate,
            fir_length,
            input_connectivity,
            input_scaling,
            rescale_input,
            ridge,
            readout_bias,
            ps: PS5000a = None,
            moku: FIRFilterBox = None,
            wfg: Agilent33220A = None,
            config=None,
            procedure=None
    ):
        self.ps = ps
        self.moku = moku
        self.wfg = wfg
        if config is  None:
            config = dict()
        for key in [
                "trigger_low",
                "trigger_high",
                "trigger_width",
                "log"
            ]:
            config[key] = config.pop(key, None)
        if config["log"] is None:
            import logging
            config["log"] = logging.getLogger('')
            config["log"].addHandler(logging.NullHandler())
        config["simulation"] = self.wfg is None or self.moku is None
        self.config = config
        self.procedure = procedure
        self.seed = seed
        self.nodes = nodes
        self.neuron_duration = neuron_duration
        self.logging_mode = logging_mode
        self.logging_rate = logging_rate
        self.fir_in_gain = fir_in_gain
        self.fir_rate = fir_rate
        self.fir_length = fir_length
        self.input_connectivity = input_connectivity
        self.input_scaling = input_scaling
        self.rescale_input = rescale_input
        self.readout_bias = readout_bias
        self.ridge = ridge
        self.trigger_level = (config["trigger_low"], config["trigger_high"])
        trigger_width = config["trigger_width"]
        if isinstance(trigger_width, str):
            trigger_width = trigger_width.replace("[", "").replace("]", "")
            trigger_width = trigger_width.split(",")
        self.trigger_width = np.atleast_1d(trigger_width).astype(float)
        self.readout = Ridge(ridge=self.ridge, input_bias=self.readout_bias)
        self.readout_capacity = Ridge(ridge=self.ridge)

        self._coeffs = None
        self.X = None
        self.transients = None
        self.Win = None
        self.Vpi = None
        self.Pmax = None
        self.M = None
        self.phi = None
        self.v_att_interp = None

        self._optical_power = optical_power
        self._phi0 = phi0

        if not self.simulation:
            self.set_clock(self.clock_rate)
            self.init_ps()
            self.init_moku()
            self.init_wfg()
        self.rng = np.random.default_rng(self.seed)

    def __getattr__(self, item):
        try:
            return self.__getattribute__(item)
        except AttributeError:
            if item in self.config:
                return self.config[item]
        raise AttributeError("ExperimentalReservoir does not have attribute {}".format(item))

    @property
    def hypers(self):
        params = {
            "optical_power": self.optical_power,
            "phi0": self.phi0,
            "nodes": self.nodes,
            "seed": self.seed,
            "neuron_duration": self.neuron_duration,
            "logging_mode": self.logging_mode,
            "logging_rate": self.logging_rate,
            "fir_in_gain": self.fir_in_gain,
            "fir_rate": self.fir_rate,
            "fir_length": self.fir_length,
            "input_connectivity": self.input_connectivity,
            "input_scaling": self.input_scaling,
            "rescale_input": self.rescale_input,
            "readout_bias": self.readout_bias,
            "ridge": self.ridge,
        }
        return params

    @property
    def calibration(self):
        result = {
            "Pmax": self.Pmax,
            "Vpi": self.Vpi,
            "phi": self.phi,
            "M": self.M,
            "ext_ratio": self.ext_ratio,
        }
        return result

    def set_clock(self, rate):
        # Use picoscope's signal generator as clock
        self.log.debug("Setting clock rate to %f Hz", rate)
        self.ps.setSigGenBuiltInSimple(0, 4, "Square", rate)

    def init_wfg(self):
        self.log.debug("Configuring burst mode")
        self.wfg.burst_state = True
        self.wfg.burst_ncycles = self.ncycles
        self.wfg.burst_mode = 'TRIG'
        self.wfg.trigger_source = 'EXT'
        self.wfg.write("BURST:INT:PERiod %f" % self.burst_period)

        self.log.debug("Setting USER waveform")
        self.wfg.shape = 'USER'
        self.wfg.frequency = self.frequency
        self.wfg.offset = 0
        self.wfg.write_binary_values(f"DATA:DAC VOLATILE,",
                                     np.zeros(1), datatype='h', is_big_endian=False)
        self.wfg.check_errors()
        self.log.debug("Setting amplitude to 10 V")
        self.wfg.amplitude = 10
        self.wfg.check_errors()
        self.log.debug("Current amplitude = %f", self.wfg.amplitude)
        self.wfg.write('VOLT:RANGe:AUTO OFF')
        self.wfg.write("FUNC:USER VOLATILE")

        self.wfg.ask('*OPC?')

        self.log.debug("Setting byte order to SWAP")
        self.wfg.write("FORMat:BORDer SWAP")

        try:
            raise_instrument_errors(self.wfg)
        except pyvisa.Error as e:
            self.log.warning(e.args[0])

    def init_scope(self):
        self.log.debug("Configuring timebase")
        self.scope.write(":TIMebase:MODE MAIN")
        self.scope.timebase_range = 2 / self.wfg.frequency
        self.scope.write(":TIMebase:DELay 0")
        self.scope.write(":TIMebase:REF {}".format(self.timebase_ref))

        self.log.debug("Configure acquisition")
        self.scope.write(":ACQuire:TYPE NORMal")

        self.log.debug("Configuring Channels 1 and 2")
        for i in range(2):
            self.scope.ch(i + 1).probe_attenuation = 1
            self.scope.ch(i + 1).display = True

        self.log.debug("Configuring Channel 1")
        self.scope.ch1.offset = self.wfg.offset
        self.scope.ch1.range = 4 * self.wfg.amplitude

        self.log.debug("Configuring Channel 2")
        self.scope.ch2.offset = 1
        self.scope.ch2.range = 10

        self.log.debug("Configuring trigger")
        self.scope.write(":TRIG:SWEep NORM")
        if self.trigger_holdoff:
            self.scope.write(":TRIG:HOLD {}".format(self.trigger_holdoff))
        self.scope.write(":TRIG:EDGE:SOUR {}".format(self.trigger_source))
        self.scope.write(":TRIG:EDGE:SLOPe POS")
        self.scope.write(":TRIG:EDGE:LEVEL {}".format(self.trigger_low))

        self.scope.write(":WAVeform:BYTeorder LSBFirst")
        self.scope.write(":WAVeform:UNSigned 0")
        self.scope.write(":WAVeform:SOURce {}".format(self.signal_channel))
        self.scope.write(":WAVeform:FORMat {}".format(self.waveform_format))
        self.scope.write(":WAVeform:POINts {}".format("MAX"))
        self.scope.write(":WAVeform:POINts:MODE {}".format('RAW'))

        self.scope.ask("*OPC?")
        raise_instrument_errors(self.scope)

    def init_ps(self):
        # TODO: parametrize
        for ch in ["A", "B"]:
            self.ps.setChannel(channel=ch, enabled=False)
        for ch in ["C", "D"]:
            self.ps.setChannel(channel=ch, coupling="DC", VRange=10)
        sample_duration = 0.1 / self.clock_rate
        sample_interval = sample_duration / 1024
        self.ps.setSamplingInterval(sample_interval, sample_duration)
        self.ps.setSimpleTrigger("D", threshold_V=0.1)

    def init_moku(self):
        self.moku.set_input_gain(1, self.fir_in_gain)
        self.moku.enable_output(1, signal=True, output=True)

        self.log.debug("Setting Moku probes")
        self.moku.set_monitor(1, "Input1")
        self.moku.set_monitor(2, "Input2")

        self.log.debug("Setting Moku scope's timebase")
        timebase_start = 0
        timebase_end = 1 / self.frequency
        self.moku.set_timebase(timebase_start, timebase_end)

        self.log.debug("Setting Moku scope's trigger")
        self.moku.set_trigger(
            type='Edge',
            source=self.trigger_source_moku,
            holdoff=self.trigger_holdoff,
            level=self.trigger_low,
            auto_sensitivity=False,
        )

    def init_Win(self, input_dim, seed=None):
        if seed is None:
            seed = self.seed
        self.Win = reservoirpy.mat_gen.uniform(
            self.nodes, input_dim, seed=seed,
            connectivity=self.input_connectivity,
            low=-1, high=1
        )

    def init_coeffs(self, seed=None):
        self._coeffs = np.zeros(self.fir_length)
        self._coeffs[-1] = 1
        self.set_fir_coeffs(self._coeffs)

    @property
    def fir_rate_float(self):
        result = float(self.fir_rate[:-3])
        if self.fir_rate[-3] == "k":
            result *= 1e3
        elif self.fir_rate[-3] == "M":
            result *= 1e6
        else:
            raise ValueError
        return result

    @property
    def frequency(self):
        return self.logging_rate / WFG_ARB_MAX_POINTS

    @property
    def optical_power(self):
        return self._optical_power

    @optical_power.setter
    def optical_power(self, val):
        self._optical_power = val
        voltage = np.interp(val, self.p_att, self.v_att)
        self.log.debug("Setting Pmax to %f mW (V_att=%.1f V))", val, voltage)
        if not self.simulation:
            info = self.moku.set_power_supply(2, True, voltage=voltage)
            self.log.debug(info)

    @property
    def phi0(self):
        return self._phi0

    @phi0.setter
    def phi0(self, val):
        self._phi0 = val
        if not self.simulation:
            Vb = self.get_bias_voltage(val)
            info = self.moku.set_power_supply(1, True, Vb)
            self.log.debug(info)

    def measure_optical_power(self):
        if self.ps is not None:
            self.ps.runBlock()
            while not self.ps.isReady():
                time.sleep(0.01)
            samples = self.ps.getDataV("C")
        else:
            samples = self.moku.get_data()["ch1"]
        Vd = np.mean(samples)
        power_mW = Vd / self.gain * 1e3
        return power_mW

    def calibrate_modulator(self, vmax):
        if not self.simulation:
            self.log.debug("Calibrating modulator")
            voltages = np.linspace(-4.5, 4.5, 15)
            powers = np.zeros_like(voltages)
            self.moku.enable_output(1, False)
            self.moku.set_power_supply(2, voltage=vmax)
            self.moku.set_power_supply(1, voltage=voltages[0])
            time.sleep(1)
            for i, vb in enumerate(tqdm(voltages)):
                self.moku.set_power_supply(1, voltage=vb)
                time.sleep(1)
                powers[i] = self.measure_optical_power()
            np.savetxt(
                os.path.join(DATA_DIR, "P_vs_Vb.csv"),
                np.column_stack((voltages, powers)),
                delimiter=",",
                header="Voltage (V), Power (mW)",
            )
            self.moku.set_power_supply(1, voltage=0)
            self.moku.enable_output(1, True)
            self.log.debug("Done calibrating modulator")
        else:
            self.log.debug("Pretending calibrating modulator")

    def calibrate_attenuator(self, vmax):
        if not self.simulation:
            self.log.debug("Calibrating attenuator")
            self.moku.enable_output(1, False)
            self.moku.set_power_supply(1, voltage=self.get_bias_voltage(np.pi / 2))
            voltages = np.linspace(vmax, 0, 15)
            powers = np.zeros_like(voltages)
            for i, v in enumerate(tqdm(voltages)):
                self.moku.set_power_supply(2, voltage=v)
                time.sleep(1)
                powers[i] = self.measure_optical_power()
            np.savetxt(
                os.path.join(DATA_DIR, "P_vs_Vatt.csv"),
                np.column_stack((voltages, powers)),
                delimiter=",",
                header="Voltage (V), Power (mW)",
            )
            self.moku.set_power_supply(2, voltage=0)
            self.moku.enable_output(1, True)
            self.log.debug("Done calibrating attenuator")
        else:
            self.log.debug("Pretending calibrating attenuator")

    def load_modulator_calibration(self):
        self.log.debug("Loading modulator calibration data from file")
        data = np.loadtxt(
            os.path.join(DATA_DIR, "P_vs_Vb.csv"),
            delimiter=",",
            skiprows=1,
        )
        voltages, powers = data.T
        Pmax, Vpi, M, phi = retrieve_mzm_params(voltages, powers)
        self.Pmax = Pmax
        self.Vpi = Vpi
        self.M = M
        self.phi = phi
        self.ext_ratio = 10 * np.log10((1 + M) / (1 - M))
        self.log.debug("Modulator calibration params:\n"
                      "Pmax=%.3f mW, Vpi=%.2f V, M=%.3f (ER=%.1f dB), phi=%.3f pi",
                      Pmax, Vpi, M, self.ext_ratio, phi / np.pi)

    def load_attenuator_calibration(self):
        self.log.debug("Loading attenuator calibration params from file")
        data = np.loadtxt(
            os.path.join(DATA_DIR, "P_vs_Vatt.csv"),
            delimiter=",",
            skiprows=1,
        )
        voltages, powers = data.T
        idx = np.argsort(voltages)
        self.v_att = voltages[idx]
        self.p_att = powers[idx]

    def get_bias_voltage(self, phi0):
        return (phi0 - self.phi) / np.pi * self.Vpi

    @property
    def coeffs(self):
        return self._coeffs

    def set_fir_coeffs(self, val):
        self._coeffs = np.asarray(val)
        if not self.simulation:
            self.log.debug("Setting Moku FIR coefficients")
            info = self.moku.set_custom_kernel_coefficients(1, self.fir_rate, self._coeffs.tolist())
            self.log.debug(info)

    def multiplex(self, x):
        waveform = x.T.ravel().repeat(self.neuron_duration)
        return waveform

    def demultiplex(self, waveform):
        nd = self.neuron_duration
        n_timesteps = len(waveform) // nd // self.nodes
        result = np.empty(n_timesteps * self.nodes)
        for i in range(len(waveform) // nd):
            result[i] = np.mean(waveform[nd * i:nd * (i + 1)])
        return result.reshape(n_timesteps, self.nodes) / self.Vpi

    def locate_transients(self, x, trig):
        distance = None
        if self.trigger_holdoff:
            distance = int(self.trigger_holdoff * self.logging_rate)
        width = np.array(self.trigger_width * self.logging_rate, dtype=int)
        if len(width) == 1:
            width = width[0]
        tstart, *stuff = scipy.signal.find_peaks(
            np.diff(trig),
            height=self.trigger_level,
            distance=distance,
            width=width,
        )
        duration = int(x.shape[0] * self.nodes * self.neuron_duration)
        idx = []
        for i in range(len(tstart)):
            idx.append(slice(tstart[i] + 1, tstart[i] + duration + 1))
        return idx

    def get_dac_samples(self, samples):
        if len(samples) >= int(WFG_ARB_MAX_POINTS):
            self.log.warning(
                "Number of samples (%d) exceeds DAC memory (%d)",
                len(samples), WFG_ARB_MAX_POINTS
            )
            samples = samples[:WFG_ARB_MAX_POINTS - 2]
        if self.rescale_input:
            max_abs = np.max(np.abs(samples))
            if max_abs != 0:
                samples /= max_abs  # Rescale data range to [-1, 1]
        if len(samples) < WFG_ARB_MAX_POINTS_FAST:
            dac_num_points = WFG_ARB_MAX_POINTS_FAST
        else:
            dac_num_points = WFG_ARB_MAX_POINTS
        dac = np.zeros(dac_num_points, dtype='h')
        dac[1:len(samples) + 1] = np.asarray(samples) * WFG_DAC_MAX_INT
        return np.clip(dac, -WFG_DAC_MAX_INT, WFG_DAC_MAX_INT)

    def generate_waveform(self, dac):
        if self.simulation:
            return
        self.wfg.output = False
        self.wfg.pulse_period = self.pulse_period
        self.wfg.write_binary_values(f"DATA:DAC VOLATILE,",
                                     dac, datatype='h', is_big_endian=False)
        self.log.debug("WFG amplitude = %f", self.wfg.amplitude)
        target_amplitude = self.input_scaling * self.Vpi
        self.log.debug("Setting WFG amplitude to %f", target_amplitude)
        self.wfg.check_errors()
        self.wfg.amplitude = target_amplitude
        self.wfg.check_errors()
        self.log.debug("WFG amplitude = %f", self.wfg.amplitude)
        self.wfg.ask("*OPC?")
        self.wfg.output = True

    def capture_waveform(self):
        if isinstance(self.scope, KeysightDSOX1102G):
            self.scope.write("*OPC")
            preamble = self.scope.ask(":WAVeform:PREamble?")
            data = self.scope.binary_values(":WAVeform:DATA?", dtype='b')
            # Parse the preamble information
            preamble_info = [int(val) for val in preamble.split(',')]
            format_code = preamble_info[0]
            y_increment = preamble_info[7]
            y_origin = preamble_info[8]
            y_reference = preamble_info[9]

            # Rescale and interpret the data
            if format_code == 1:  # WORD format
                data = np.frombuffer(data, dtype=np.int16)
                data = y_origin + (data - y_reference) * y_increment
            elif format_code == 0:  # BYTE format
                data = np.frombuffer(data, dtype=np.int8)
                data = y_origin + (data - y_reference) * y_increment

            return data
        else:
            data = self.moku.get_data()
            return data

    def fit(self, x, y, run=True, states=None):
        if run:
            states = self.run(x)
        elif states is None:
            states = self.state
        self.readout.fit(states, y)
        return self

    @property
    def state(self):
        if self.transients is None:
            return None
        result = np.concatenate(self.transients["state"])
        split_idx = np.cumsum([len(_) for _ in self.x[:-1]])
        result = np.split(result, split_idx)
        return result

    def download_history(self):
        self.log.debug("Downloading history from Moku device")
        target_name, target_ext = os.path.splitext(self.history_path)
        li_file = os.path.join(os.path.dirname(self.history_path), self.datalogger["file_name"])
        self.moku.download("persist", self.datalogger["file_name"], li_file)
        self.moku.delete("persist", self.datalogger["file_name"])
        self.log.debug("Converting .li file to {}".format(target_ext))
        subprocess.check_call(["mokucli", "convert", li_file, "--format={}".format(target_ext[1:])])
        os.remove(li_file)
        converted = li_file.replace(".li", target_ext)
        os.replace(converted, self.history_path)

    def run(self, x):
        self.load_modulator_calibration()
        self.load_attenuator_calibration()
        self.optical_power = self._optical_power
        self.phi0 = self._phi0
        if self.Win is None:
            self.init_Win(x[0].shape[-1])
        if self._coeffs is None:
            self.init_coeffs()
        self.x = x
        self.X = join_items(x, self.config["batch"])

        self.transients = defaultdict(list)
        for i in range(len(self.X)):
            if self.procedure:
                if self.procedure.should_stop():
                    return
            if self.simulation:
                transient = self.simulate_transient(self.X[i])
            else:
                self.log.debug("Sending batch %d of %d", i + 1, len(self.X))
                self.dac_samples = self.get_dac_samples(
                    self.multiplex(self.Win.dot(self.X[i].T))
                )
                print('the len of x is:', len(self.X[i]))
                print('The ratio between dac_samples and max samples', len(self.dac_samples) / WFG_ARB_MAX_POINTS)
                if len(self.dac_samples)>len(self.X[i])*self.nodes:
                    print('the len of dac_samples is:', len(self.dac_samples),'bigger')
                self.pulse_period = self.dac_samples.shape[0] / self.logging_rate
                self.generate_waveform(self.dac_samples)
                self.wfg.write("DISP:TEXT 'Batch {}/{}'".format(i + 1, len(self.X)))
                if self.scope_type == "pico":
                    transient = self.detect_transient_using_pico(self.X[i])
                else:
                    transient = self.detect_transient_using_moku(self.X[i])
                self.wfg.output = False
            for k, v in transient.items():
                self.transients[k].append(v)
        return self.state

    def simulate_transient(self, x):
        G = 10 * self.optical_power / self.Vpi
        noise = 10 * (2 ** -12) / self.Vpi
        M = self.M
        Phi0 = self.phi0
        feedback_ratio = 0.5
        beta = feedback_ratio
        rho = (1 - feedback_ratio) * self.input_scaling
        freq_ratio = round(self.fir_rate_float / self.logging_rate)
        u = self.Win.dot(x.T).T.ravel().repeat(self.neuron_duration)
        u = scipy.signal.resample(u, len(u) * freq_ratio)
        N = x.shape[0] * self.nodes
        h = self.coeffs
        L = len(h)
        s = np.zeros((N * freq_ratio + L))
        s[:L] = scipy.optimize.fixed_point(mzm_func, 0, (G, M, Phi0))
        for n in range(N * freq_ratio):
            s[n + L] = mzm_func(beta * h.dot(s[n + L:n:-1]) + rho * u[n], G, M, Phi0)
            s[n + L] += self.rng.uniform(-noise / 2, noise / 2)
        s = scipy.signal.decimate(s[L:], freq_ratio)
        transient = dict()
        transient["raw"] = s
        transient["state"] = s.reshape(x.shape[0], self.nodes)
        return transient

    def detect_transient_using_pico(self, x):
        nsamples = int(x.shape[0] * self.nodes * self.neuron_duration)
        oversampling_ratio = 10
        sample_duration = nsamples / self.logging_rate
        sample_interval = sample_duration / nsamples / oversampling_ratio
        self.ps.setSamplingInterval(sample_interval, sample_duration)
        time.sleep(1 / self.clock_rate)
        self.ps.runBlock()
        while not self.ps.isReady():
            time.sleep(0.01)
        raw = self.ps.getDataV("C")
        tscope = np.linspace(0, self.ps.sampleInterval * self.ps.noSamples, self.ps.noSamples)
        t = np.linspace(0, sample_duration, nsamples)
        data = np.interp(t, tscope, raw)
        transient = dict()
        transient["raw"] = raw
        transient["resampled"] = data
        transient["state"] = self.demultiplex(data)
        return transient

    def _detect_transient_using_moku(self, x):
        log_duration = self.config["log_duration"]
        clock_rate = self.clock_rate
        self.datalogger = self.moku.start_logging(
            duration=log_duration,
            mode=self.logging_mode,
            file_name_prefix=self.logging_fn_prefix,
            rate=self.logging_rate
        )
        while not self.moku.logging_progress()["complete"]:
            time.sleep(0.1)
        self.download_history()
        self.log.debug("Loading history from {}".format(self.history_path))
        target_name, target_ext = os.path.splitext(self.history_path)
        if target_ext == '.csv':
            history = np.loadtxt(self.history_path, delimiter=",", comments="%", skiprows=1)
        elif target_ext in [".hdf5", ".h5"]:
            with h5py.File(self.history_path, "r") as f:
                history = f["moku"][:]
        else:
            history = np.load(self.history_path)
        t = history["Time (s)"]
        ch1 = history["Probe A (V)"]
        ch2 = history["Probe B (V)"]
        dt = t[1] - t[0]
        self.log.debug("Locating transients in history")
        idx = self.locate_transients(x, ch2)
        self.log.debug("Done")
        expected_num_transients = clock_rate * log_duration + 1
        self.log.debug("Detected %d transients (expected: less or equal to %d)", len(idx), expected_num_transients)
        if len(idx) > expected_num_transients:
            raise ValueError
        transient = dict()
        transient["ch1"] = np.mean([ch1[_] for _ in idx[:-1]], axis=0)
        transient["ch2"] = np.mean([ch2[_] for _ in idx[:-1]], axis=0)
        transient["time"] = dt * np.arange(len(ch1[idx[0]]))
        transient["state"] = self.demultiplex(transient["ch1"])
        return transient

    def detect_transient_using_moku(self, x):
        attempts = self.config["attempts"]
        for attempt in range(attempts):
            if self.procedure:
                if self.procedure.should_stop():
                    return
            if attempt > 0:
                self.log.debug(
                    "Retrying to detect transients (attempt %d/%d)", attempt + 1, attempts
                )
            try:
                transient = self._detect_transient_using_moku(x)
                return transient
            except ValueError:
                if attempt > attempts // 2:
                    self.log.warning("Failed to detect transients in %d/%d attempts", attempt, attempts)
        raise ValueError("Failed to detect transients with %d attempts", attempts)

