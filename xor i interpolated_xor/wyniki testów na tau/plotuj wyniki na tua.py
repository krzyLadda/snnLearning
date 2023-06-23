from matplotlib.colors import LogNorm
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
mpl.rcParams['lines.markersize'] = 2.5
plt.rcParams.update({
    "text.usetex": True,
    # "font.family": "sans-serif",
    # "font.sans-serif": "Helvetica",
    'font.size': 11,
})
plt.rc('lines', linewidth=0.7)

data = pd.read_excel("plot.ods", engine="odf")
data = data.iloc[::-1]
rate = [x for x in data.columns[1:]]
tau = data['tau'].tolist()

values = {}
for row, t in zip(data.iloc, tau):
    values[t] = row[1:].tolist()

plt.figure(1)
markers = ["o", "v", "x", "D", "+", "s", "*", "^"]
ax = plt.gca()
ax.set_yscale('log')
ax.set_xscale('log')
for i, t in enumerate(tau):
    l = r'$\tau_s = '+str(t)+r'$'
    plt.plot(rate, values[t], marker=markers[i], label=l)
    for i, v in enumerate(values[t]):
        if v == '-' or np.isnan(v):
            continue
        # ax.text(rate[i], v, v, size=8)

ax.legend(ncol=int(len(tau)/2), loc='lower left', bbox_to_anchor=(0.0, 1.0), fancybox=True)

foo = np.array(list(values.values()))
plt.ylim([0.95*np.nanmin(foo), 1.05*np.nanmax(foo)])
plt.xlim([0.95*rate[0], 1.05*rate[-1]])

ax.set_xticks(rate)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

yticks = list(range(30, 100, 10)) + list(range(100, 476, 50))
ax.set_yticks(yticks)
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xticks(rotation=45)
#plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

plt.grid(linewidth=0.4)
plt.xlabel(r"Learning rate $\eta$")
plt.ylabel(r"Average number of epoch to convergence")
plt.show()

# heat map
fig, ax = plt.subplots()
d = data.to_numpy()
d = d[:, 1:]
d = np.delete(d, 6, 0)
im = ax.imshow(d, norm=LogNorm(vmin=30, vmax=230))

bar_ticks = list(range(30, 75, 5)) + list(range(50, 231, 50))
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(label=r'Average number of epochs to convergence', size=13)
cbar.set_ticks(bar_ticks)
cbar.set_ticklabels(bar_ticks)
tau = [x for x in tau if x != 5]
ax.set_xticks(np.arange(len(rate)), labels=rate)
ax.set_yticks(np.arange(len(tau)), labels=tau)
ax.set_xlabel(r"Learning rate $\eta$", size=16)
ax.set_ylabel(r"Time constant of synapses $\tau_s$", size=16)


for i in range(len(tau)):
    for j in range(len(rate)):
        text = ax.text(j, i, d[i, j],
                       ha="center", va="center", color="w")

plt.show()
