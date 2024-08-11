import argparse
import os
import time
import numpy as np
import pyvisa
import reservoirpy
from picoscope.ps5000a import PS5000a
from moku.instruments import FIRFilterBox
from pymeasure.instruments.agilent import Agilent33220A
from pymeasure.experiment import (
    Procedure,
    Metadata,
    BooleanParameter,
    IntegerParameter,
    FloatParameter,
    ListParameter,
    Parameter,
)
from pymeasure.log import console_log

from config import WFG_ADDRESS_PATTERN, DATA_DIR
from instruments import moku_list
from reservoir import ExperimentalReservoir

import logging
log = logging.getLogger('')
log.addHandler(logging.NullHandler())


class ProcedureBase(Procedure):

    simulation = BooleanParameter("Simulation", default=False)

    iterations = IntegerParameter("Max. Iterations", minimum=1, default=100)
    clock_rate = FloatParameter("Clock Rate", units='Hz', minimum=1, default=5, maximum=5e6)

    measure_power = BooleanParameter("Measure Power", default=False)

    ncycles = IntegerParameter("Number of Bursts", minimum=1, default=1)
    input_scaling = FloatParameter("Input Scaling", default=1, decimals=2)
    neuron_duration = IntegerParameter("Neuron Duration", minimum=1, default=1)

    input_connectivity = FloatParameter("Input Connectivity", default=0.1, decimals=2)
    signal_channel = Parameter("Scope Signal Channel", default="CHAN1")
    trigger_source = Parameter("Scope Trigger Source", default="EXT")
    signal_channel_moku = Parameter("Moku Signal Channel", default="ch1")
    burst_period = FloatParameter("Burst Period", default=0.1)
    trigger_source_moku = Parameter("Moku Trigger Source", default="ProbeB")
    trigger_low = FloatParameter("Trigger Low Level", units='V', default=1)
    trigger_high = FloatParameter("Trigger High Level", units='V', default=4)
    trigger_holdoff = FloatParameter("Trigger Holdoff", units='s', default=0)
    trigger_width = Parameter("Trigger Pulse Width", default="0")
    attempts = IntegerParameter("Number of Attempts", default=50)
    log_duration = IntegerParameter("Log duration", default=1, units='s')

    timebase_ref = Parameter("Timebase Reference", default='CENTer')
    nodes = IntegerParameter("Number of reservoir nodes", default=100)
    fir_in_gain = FloatParameter("FIR input gain", default=0, units="dB")
    fir_length = IntegerParameter("FIR kernel length", default=232)
    fir_rate = ListParameter(
        "FIR sampling rate",
        choices=[
            "3.906MHz",
            "1.953MHz",
            "976.6kHz",
            "488.3kHz",
            "244.1kHz",
            "122.1kHz",
            "61.04kHz",
            "30.52kHz",
        ],
        default="3.906MHz",
    )
    ext_in_vmax = FloatParameter("Ext. In Max Voltage", default=5, units='V')
    optical_power = FloatParameter("Optical Power", default=0.4, units='mW')
    phi0 = FloatParameter("Phase Bias", default=0, minimum=-np.pi, maximum=np.pi, decimals=2)
    gain = ListParameter("Photodetector Gain", choices=[1e3, 1e4, 1e5, 1e6], default=1e4)

    transients_path = Parameter("Transients Path",
                                default=os.path.join(DATA_DIR, "transients.pickle"))
    dataset_dump_path = Parameter("Dataset Dump Path",
                                default=os.path.join(DATA_DIR, "dataset.pickle"))
    weights_dump_path = Parameter("Weights Dump Path",
                                default=os.path.join(DATA_DIR, "weights.pickle"))

    batch = IntegerParameter("Batch Size", default=4, minimum=1)
    seed = IntegerParameter("Random Seed", default=1)
    ridge = FloatParameter("Regularization Constant", default=0)
    readout_bias = BooleanParameter("Enable readout bias", default=True)
    rescale_input = BooleanParameter("Rescale input samples to [-1, 1] range", default=True)

    waveform_format = Parameter("Waveform Format", default='BYTE')

    scope_type = ListParameter(
        "Scope Type",
        choices=[
            "moku",
            "pico",
        ],
        default="moku"
    )
    logging_mode = Parameter("Datalogger Logging Mode", default="Normal")
    logging_fn_prefix = Parameter("Datalogger Filename Prefix", default="rc_transients")
    logging_rate = IntegerParameter("Datalogger Rate", default=488.3e3, units="Sa/s")
    history_path = Parameter("History Path", default="data/history.hdf5")

    verbosity = IntegerParameter("Console Log Verbosity", default=logging.INFO)

    starttime = Metadata('Start time', fget=time.time)

    log = None
    wfg: Agilent33220A = None
    moku: FIRFilterBox = None
    ps: PS5000a = None
    reservoir: ExperimentalReservoir = None
    M = None
    Vpi = None
    phi = None
    Pmax = None

    def start_wfg(self):
        if self.simulation:
            return
        log.info("Initializing waveform generator")
        rm = pyvisa.ResourceManager()
        address = rm.list_resources(WFG_ADDRESS_PATTERN)[0]
        log.info("Found waveform generator {}".format(address))
        self.wfg = Agilent33220A(address)
        log.info("Connected to {}".format(self.wfg.id))
        self.wfg.reset()
        self.wfg.clear()
        self.wfg.write("*CLS")
        self.wfg.ask("*OPC?")

    def start_moku(self):
        if self.simulation:
            return
        device = moku_list().pop()
        moku_ip = "[" + device["IP"] + "]"
        log.info("Connecting to the Moku device at {}".format(moku_ip))
        self.moku = FIRFilterBox(moku_ip, force_connect=True, ignore_busy=True, read_timeout=300)

    def start_ps(self):
        if self.simulation:
            return
        self.ps = PS5000a()
        log.info("Found the following picoscope: %s", self.ps.getAllUnitInfo())

    def init_reservoir(self, **kwargs):
        reservoir_params = [
            "optical_power",
            "phi0",
            "nodes",
            "seed",
            "neuron_duration",
            "logging_mode",
            "logging_rate",
            "fir_in_gain",
            "fir_rate",
            "fir_length",
            "input_connectivity",
            "input_scaling",
            "rescale_input",
            "ridge",
            "readout_bias",
        ]
        config_params = [
            "logging_fn_prefix",
            "history_path",
            "transients_path",
            "dataset_dump_path",
            "weights_dump_path",
            "log",
            "clock_rate",
            "burst_period",
            "trigger_low",
            "trigger_high",
            "trigger_holdoff",
            "trigger_width",
            "gain",
            "attempts",
            "log_duration",
            "scope_type",
            "trigger_source",
            "trigger_source_moku",
            "ncycles",
            "batch",
        ]
        params = dict()
        for param in reservoir_params:
            params[param] = kwargs.pop(param, getattr(self, param))
        if kwargs:
            raise ValueError("Invalid arguments: {}".format(kwargs))
        config = dict()
        for param in config_params:
            config[param] = getattr(self, param)
        reservoir = ExperimentalReservoir(
                ps=self.ps,
                moku=self.moku,
                wfg=self.wfg,
                procedure=self,
                config=config,
                **params
        )
        return reservoir

    def shutdown_wfg(self):
        try:
            self.wfg.output = False
            self.wfg.check_errors()
            if self.status == self.FAILED:
                msg = "FAILED"
                for _ in range(10):
                    self.wfg.beep()
            elif self.should_stop():
                msg = "ABORTED"
            else:
                self.wfg.beep()
                msg = "COMPLETED"
            self.wfg.write("DISP:TEXT '{}'".format(msg))
            time.sleep(1)
            self.wfg.write("DISP:TEXT:CLEar")
            self.wfg.shutdown()
        except AttributeError:
            pass

    def shutdown_moku(self):
        try:
            self.moku.stop_logging()
            self.moku.stop_streaming()
            self.moku.enable_output(1, False)
        except Exception:
            pass
        try:
            self.moku.relinquish_ownership()
        except AttributeError:
            pass

    def shutdown_ps(self):
        try:
            self.ps.stop()
            self.ps.close()
        except AttributeError:
            pass

    def startup(self):
        console_log(log, self.verbosity)
        reservoirpy.verbosity(0)
        self.log = log
        self.start_ps()
        self.start_moku()
        self.start_wfg()

    def calibrate(self):
        self.reservoir.calibrate_modulator(self.ext_in_vmax)
        self.reservoir.load_modulator_calibration()
        self.reservoir.calibrate_attenuator(self.ext_in_vmax)
        self.reservoir.load_attenuator_calibration()

    def execute(self):
        raise NotImplementedError

    def shutdown(self):
        self.shutdown_wfg()
        self.shutdown_moku()
        self.shutdown_ps()
        log.info("Experiment finished")

    @classmethod
    def make_argparser(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--iterations", type=int,
                            default=cls.iterations.default)
        parser.add_argument("--simulation", action="store_true",
                            default=cls.simulation.default)
        parser.add_argument("--nodes", type=int,
                            default=cls.nodes.default)
        parser.add_argument("--ext-in-vmax", type=float,
                            default=cls.ext_in_vmax.default)
        parser.add_argument("--optical-power",
                            default=cls.optical_power.default)
        parser.add_argument("--phi0", type=float,
                            default=cls.phi0.default)
        parser.add_argument("--clock-rate", type=float,
                            default=cls.clock_rate.default)
        parser.add_argument("--signal-channel",
                            default=cls.signal_channel.default)
        parser.add_argument("--signal-channel-moku",
                            default=cls.signal_channel_moku.default)
        parser.add_argument("--burst-period", type=float,
                            default=cls.burst_period.default)
        parser.add_argument("--trigger-source",
                            default=cls.trigger_source.default)
        parser.add_argument("--trigger-source-moku",
                            default=cls.trigger_source_moku.default)
        parser.add_argument("--trigger-low", type=float,
                            default=cls.trigger_low.default)
        parser.add_argument("--trigger-high", type=float,
                            default=cls.trigger_high.default)
        parser.add_argument("--trigger-holdoff", type=float,
                            default=cls.trigger_holdoff.default)
        parser.add_argument("--trigger-width", nargs="+",
                            default=cls.trigger_width.default)
        parser.add_argument("--input-scaling", type=float,
                            default=cls.input_scaling.default)
        parser.add_argument("--burst-ncycles", type=int,
                            default=cls.ncycles.default)
        parser.add_argument("--neuron-duration", type=int,
                            default=cls.neuron_duration.default)
        parser.add_argument("--batch", type=int,
                            default=cls.batch.default)
        parser.add_argument("--seed", type=int,
                            default=cls.seed.default)
        parser.add_argument("--scope-type", choices=cls.scope_type.choices,
                            default=cls.scope_type.default)
        parser.add_argument("--logging-rate", type=float,
                            default=cls.logging_rate.default)
        parser.add_argument("--fir-in-gain", type=float,
                            default=cls.fir_in_gain.default)
        parser.add_argument("--fir-rate", choices=cls.fir_rate.choices,
                            default=cls.fir_rate.default)
        parser.add_argument("--transients-path",
                            default=cls.transients_path.default)
        parser.add_argument("--history-path", 
                            default=cls.history_path.default)
        parser.add_argument("--attempts", type=int,
                            default=cls.attempts.default)
        parser.add_argument("--log-duration", type=int,
                            default=cls.log_duration.default)
        parser.add_argument("--verbosity", type=int,
                            default=cls.verbosity.default)
        return parser

    @classmethod
    def init_procedure(cls, args):
        procedure = cls()
        for k, v in vars(args).items():
            setattr(procedure, k, v)
        return procedure

