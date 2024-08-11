import os
import sys
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt


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
    if len(sys.argv) > 2:
        raise RuntimeError("Wrong number of arguments")
    if len(sys.argv) == 1:
        csv_file = "data/P_vs_Vb.csv"
    else:
        csv_file = sys.argv[1]
    os.chdir(os.path.dirname(__file__))

    data = np.loadtxt(csv_file, np.double, skiprows=1, delimiter=',')
    voltages, powers = data.T
    # print("Experimental Vb = ", voltages)
    # print("Experimental Pout = ", powers)
    params = retrieve_mzm_params(voltages, powers)
    Pmax, Vpi, M, phi = params
    Pmin = Pmax / 2 * (1 - np.abs(M))
    print(f"Pmax = {Pmax:0f} mW")
    print(f"Pmin = {Pmin:0f} mW")
    print(f"Ext. Ratio = {10 * np.log10(Pmax / Pmin):0f} dB")
    print(f"Vpi = {Vpi:1f} V")
    print(f"M = {M:.3f}")
    print(f"phi = {phi / np.pi:.2f}pi")

    # ext_ratio = 10 * np.log10(Pmax / Pmin)
    # print(f"Extinction ratio @ DC = {ext_ratio:.1f} dB")


    fig, ax = plt.subplots(
        figsize=(5, 4),
        # dpi=300,
        layout='tight'
    )
    ax.plot(voltages, powers, 'k.', label="Measured")
    vs = np.linspace(min(voltages), max(voltages), 100)
    ax.plot(vs, power_func(vs, *params), 'r-', label='Fitted')
    ax.set_xlabel(r"$V_B$ [V]")
    ax.set_ylabel(r"$P$ [mW]")
    ax.set_ylim(-0.1 * Pmax, 1.1 * Pmax)
    ax.grid(True)
    ax.legend()
    # ax.annotate(
        # r'$P(V)=(P_\mathrm{max}-P_\mathrm{min})\cos^2(\pi \frac{V}{2V_{\pi}}+\varphi)+P_\mathrm{min}$',
        # (0, 1.4 * np.mean(powers)),
        # size=8
    # )
    # ax.annotate(
    #     f'$V_\pi$={Vpi:.1f} V',
    #     (4, 1. * np.mean(powers)),
    #     size=10
    # )
    # ax.set_title("MZM's $V_\pi$ measurements in DC regime")
    fig.savefig("Pout_vs_Vb.png", dpi=300, facecolor='white')
    plt.show()
