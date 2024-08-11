import os
import sys
import numpy as np
from scipy.optimize import curve_fit
import sympy
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rc("font", family="Arial")
plt.rc("mathtext", fontset="cm")


def power_func(Vb, Pmax, Vpi, M, phi):
    return Pmax / 2 * (1 + M * np.sin(np.pi / Vpi * Vb + phi))


def retrieve_mzm_params(voltages, powers):
    params, covariance = curve_fit(
        power_func, voltages, powers,
        p0=(max(powers), 5, 1, 0),
        bounds=[[0, 0, 0, -np.pi], [np.inf, 10, 1, np.pi]]
    )
    Pmax, Vpi, M, phi = params
    return Pmax, Vpi, M, phi


if __name__ == "__main__":
    if len(sys.argv) > 3 or len(sys.argv) == 2:
        raise RuntimeError("Wrong number of arguments")
    elif len(sys.argv) == 3:
        csv_file_bias, csv_file_att = sys.argv[1:]
    else:
        csv_file_bias = "data/P_vs_Vb.csv"
        csv_file_att = "data/P_vs_Vatt.csv"
    os.chdir(os.path.dirname(__file__))
    
    data = np.loadtxt(csv_file_bias, np.double, skiprows=1, delimiter=',')
    Vb, Pb = data.T
    data_att = np.loadtxt(csv_file_att, delimiter=',', skiprows=1)
    Vatt, Patt = data_att.T
    # print("Experimental Vb = ", voltages)
    # print("Experimental Pout = ", powers)
    params = retrieve_mzm_params(Vb, Pb)
    Pmax, Vpi, M, phi = params
    Pmin = Pmax / 2 * (1 - np.abs(M))

    print(f"Pmax = {Pmax:0f} mW")
    print(f"Pmin = {Pmin:0f} mW")
    print(f"M = {M:.3f} (ext. ratio = {10 * np.log10(Pmax / Pmin):0f} dB)")
    print(f"Vpi = {Vpi:1f} V")
    print(f"M = {M:.3f}")
    print(f"phi = {phi / np.pi:.2f}pi")

    config = dict(figsize=(8, 3), layout="tight")
    fig, (ax1, ax2) = plt.subplots(1, 2, **config)

    Vb_dense = np.linspace(min(Vb), max(Vb), 100)
    xdata, ydata = np.meshgrid(Vb_dense, Patt)
    zdata = power_func(xdata, ydata, Vpi, M, phi)
    xdata, ydata = np.meshgrid(Vb_dense, Vatt)

    ax1.plot(Vb_dense, power_func(Vb_dense, *params), 'r-', label='Fitted')
    ax1.plot(Vb, Pb, 'ko', label="Measured")
    ax1.set_xlabel(r"Bias voltage $V_b$ [V]")
    ax1.set_ylabel(r"Optical Power [mW]")
    ax1.grid(True)

    att = -10 * np.log10(Patt / Pmax)
    ax2.plot(Vatt, Patt, "r-", label="Interpolated")
    ax2.plot(Vatt, Patt, "ko", label="Measured")
    ax2.set_xlabel(r"Power Control Voltage [V]")
    ax2.set_ylabel("Optical Power [mW]")
    ax2.grid(True)

    ax3 = ax1.twinx()
    ax3.plot(Vb_dense, Vb_dense / Vpi + phi / np.pi, "b")
    yticks = [0, 0.5, 1, 1.5]
    labels = [r'${}$'.format(sympy.latex(sympy.pi * tick)) for tick in yticks]
    ax3.set_yticks(yticks, labels, color='b')
    ax3.set_xlabel(r"Bias voltage [V]")
    ax3.set_ylabel(r"Phase Bias $\Phi_0$", color='b')
    ax3.grid(False)
    plt.show()
