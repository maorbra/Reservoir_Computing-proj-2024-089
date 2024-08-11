import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as spopt


def mzm_func(x, G, M, Phi0):
    return 0.5 * G * (1 + M * np.sin(np.pi * x + Phi0))


def iterate_map(f, x0, *args, nsteps=500):
    n = 0
    x = x0
    while n < nsteps:
        x = f(x, *args)
        yield x
        n += 1


def fn(f, n):
    """Convert map `f` to `f^n`"""
    def wrapper(x, *args):
        return list(iterate_map(f, x, nsteps=n, *args))[-1]
    return wrapper
        

def plot_cobweb(f, x0=0, inc=1, nsteps=100, span=1, args=(), bipolar=False):
    start = -span if bipolar else 0
    identity = np.linspace(start, span, 500)
    fig, ax = plt.subplots(figsize=(3.3, 3), dpi=200)
    ax.plot(identity, identity, 'k-', lw=0.75)
    finc = fn(f, inc)
    y = finc(identity, *args)
    ax.plot(identity, y, 'k-')
    x0 = np.atleast_1d(x0)
    px, py = np.empty((2, nsteps + 1, x0.shape[0]))
    px[0], py[0] = x0, 0
    for n in range(1, nsteps, 2):
        px[n] = px[n-1]
        py[n] = finc(px[n-1], *args)
        px[n+1] = py[n]
        py[n+1] = py[n]
    ax.plot(px[1:], py[1:], linewidth=0.5)
    ax.set_xlabel('$x_n$')
    ax.set_ylabel(f'$x_{{n+{inc}}}$')
    ax.set_xlim(start, span)
    ax.set_ylim(start, span)
    ax.plot(x0, finc(x0, *args), 'b.')
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    G = 1
    M = 0.98
    Phi0 = 0
    plot_cobweb(mzm_func, 0, args=(G, M, Phi0))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    Gs = np.linspace(0, 2, 50)
    Phi0s = np.linspace(0, np.pi, 50)
    xs = np.linspace(0, 1, 2)
    fps = np.full((len(Gs), len(Phi0s), len(xs)), np.nan)
    for i in range(len(Gs)):
        for j in range(len(Phi0s)):
            try:
                fps[i, j] = spopt.fixed_point(fn(mzm_func, 2), xs, (Gs[i], M, Phi0s[j]))
            except RuntimeError:
                pass
    options = dict(color="k", marker='.', s=1, plotnonfinite=True)
    xx, yy = np.meshgrid(Gs, Phi0s)
    for i in range(fps.shape[-1]):
        ax.scatter(xx, yy, fps[:, :, i], **options)
    plt.show()