import os
import sys
import numpy as np
from scipy.stats import linregress
from matplotlib import pyplot as plt


plt.rc("font", family="Arial")
plt.rc("mathtext", fontset="cm")


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    if len(sys.argv) == 1:
        csv_name = "V_vs_P.csv"
    elif len(sys.argv) != 2:
        raise RuntimeError("Wrong number of arguments")
    else:
        csv_name = sys.argv[1]

    csv_path = os.path.join('..', 'data', csv_name)
    if not os.path.isfile(csv_path):
        print("File: {} not found".format(csv_path))
        raise FileNotFoundError
    print("Filepath: {}".format(csv_path))
    data = np.loadtxt(csv_path, np.double, skiprows=1, delimiter=',')
    Pin, Pout, Vmon, *Vgain = data.T
    
    result = linregress(Pout * 1e-3, Vgain[0])
    Gstar = result.slope
    print("Gstar =", Gstar, "[V / W]")
    fig, ax = plt.subplots(figsize=(5, 3), layout="tight")
    idx = slice(1, None, None)
    ax.plot(Pout[idx], (Vmon / Pout * 1e3)[idx], 'k', label='Monitor')
    gains = [1e3, 1e4, 1e5]
    with np.errstate(all='ignore'):
        for gain, V in zip(gains, Vgain):
            ax.plot(Pout[idx], (V / Pout)[idx] * 1e3, label=f'RF $10^{np.log10(gain):.0f}$')
    ax.set_xlabel('$P_\mathrm{out}$ [mW]')
    ax.set_ylabel('$G^*$ [V / W]')
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True)
    # axr = ax.twinx()
    # axr.set_ylim(0, ax.get_ylim()[-1] / ax.get_xlim()[-1])
    fig.savefig('../../Figures/G_vs_P.pdf', dpi=300, facecolor='white', bbox_inches='tight')
    plt.show()
