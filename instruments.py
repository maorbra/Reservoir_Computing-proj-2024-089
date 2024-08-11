import subprocess
import time
import matplotlib.pyplot as plt
from moku.instruments import FIRFilterBox
import pyvisa
from reservoirpy import mat_gen


AWG_SAMPLE_RATES = [
    'Auto',
    '1.25Gs',
    '1Gs',
    '625Ms',
    '500Ms',
    '312.5Ms',
    '250Ms',
    '125Ms',
    '62.5Ms',
    '31.25Ms',
    '15.625Ms'
]

FIR_SAMPLE_RATES = [
    '3.906MHz',
    # '1.953MHz', # Not working
    '976.6kHz',
    '488.3kHz',
    '244.1kHz',
    '122.1kHz',
    '61.04kHz',
    '30.52kHz',
]


def raise_instrument_errors(inst):
    errors = inst.check_errors()
    if not errors:
        return
    raise pyvisa.Error(errors)


def moku_list():
    p = subprocess.run(["moku", "list"], capture_output=True, bufsize=0)
    lines = p.stdout.decode('utf-8').split("\n")
    header = None
    devices = []
    while lines:
        line = lines.pop(0)
        if line.startswith("Moku Client Version"):
            continue
        if line.startswith("Name"):
            header = line
        if line.startswith('Moku'):
            devices.append(dict(zip(header.split(), line.split())))
    return devices


def run_fir_as_scope(fir: FIRFilterBox):
    # Set up the plotting parameters
    plt.ion()
    plt.show()
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.grid(True)
    ax.set_ylim([-1.2, 1.2])
    line1, = ax.plot([])
    line2, = ax.plot([])

    while True:
        if not plt.fignum_exists(1):
            break
        data = fir.get_data()

        t = data['time']
        ax.set_xlim([t[0], t[-1]])
        # Update the plot
        line1.set_data(t, data['ch1'])
        line2.set_data(t, data['ch2'])
        plt.pause(0.01)
        time.sleep(0.02)


if __name__ == '__main__':

    devices = moku_list()
    if devices:
        device = devices[0]
        print("Found Moku device:")
        print(device)
    else:
        print("No Moku devices found")

    ip = "[{}]".format(device['IP'])
    print("Connecting to the device {} at {}".format(device['Name'], ip))
    # i = MultiInstrument(ip, platform_id=2, force_connect=True, ignore_busy=True)
    fir = FIRFilterBox(ip, force_connect=True, ignore_busy=True)
    print("Connected")
    connections = [
        dict(source="Input1", destination="Slot2InA"),
        # dict(source="Slot1OutA", destination="Slot2InA"),
        dict(source="Slot2OutA", destination="Output1"),
        dict(source="Slot1OutA", destination="Output2"),
    ]

    # awg = i.set_instrument(1, ArbitraryWaveformGenerator)
    # fir = i.set_instrument(2, FIRFilterBox)

    # lut_data = np.zeros(1024)
    # sequence = np.array([1]).repeat(4)
    # lut_data[:len(sequence)] = sequence
    # awg.generate_waveform(1, 'Auto', lut_data.tolist(), frequency=1e3, amplitude=1)

    # coeffs = np.loadtxt('data/fir_coeffs.csv')
    # coeffs = np.zeros(232)
    # coeffs[20] = 0.5
    # coeffs[200] = 0.5
    coeffs = mat_gen.bernoulli(232, p=0.5, connectivity=0.02, seed=42)
    # coeffs /= np.linalg.norm(coeffs)
    fir.set_custom_kernel_coefficients(1, '3.906MHz', coeffs.tolist())
    fir.enable_output(1, signal=True, output=True)

    # Configure scope probes
    fir.set_monitor(1, "Output1")
    fir.set_monitor(2, "Input1")

    # Set timebase
    fir.set_timebase(-1e-4, 1e-4)

    # Set trigger
    fir.set_trigger(type='Edge', source='ProbeB', level=0.4)

    # Set power supply for bias
    fir.set_power_supply(1, voltage=0)
