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
        if np.isnan(v):
            continue
        # ax.text(rate[i], v, v, size=8)

ax.legend(ncol=int(len(tau)/2), loc='lower left', bbox_to_anchor=(0.0, 1.0), fancybox=True)

foo = np.array(list(values.values()))
plt.ylim([0.95*np.nanmin(foo), 1.05*np.nanmax(foo)])
plt.xlim([0.95*rate[0], 1.05*rate[-1]])

ax.set_xticks(rate)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

# + list(range(50, 121, 10)) # + [22]  # + list(range(49, 401, 25))
yticks = list(range(45, 126, 5))
ax.set_yticks(yticks)
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

plt.grid(linewidth=0.4)
plt.xlabel("Learning rate")
plt.ylabel("Average number of epoch to convergence")
plt.show()


# heat map
data2 = pd.read_excel("plot ile zbieglo.ods", engine="odf")
data2 = data2.iloc[::-1]
d = data.to_numpy()
d = d[:, 1:]

d2 = data2.to_numpy()
d2 = d2[:, 1:]

fig, ax = plt.subplots()
im = ax.imshow(d, norm=LogNorm(vmin=np.min(d), vmax=np.max(d)))

d_min = np.min(d)
d_max = np.max(d)
step = 5
if d_max < 100:
    bar_ticks = [d_min] + list(range(int(d_min - d_min % step + step),
                                     int(d_max - d_max % step + step), step)) + [np.max(d)]
else:
    step2 = 10
    bar_ticks = [d_min] + list(range(int(d_min - d_min % step + step), 100, step)) +\
        list(range(100, int(d_max - d_max % step2 + step2), step2)) + [np.max(d)]
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(label=r'Average number of epochs to convergence', size=13)
cbar.set_ticks(bar_ticks)
cbar.set_ticklabels(bar_ticks)
ax.set_xticks(np.arange(len(rate)), labels=rate)
ax.set_yticks(np.arange(len(tau)), labels=tau)
ax.set_xlabel(r"Learning rate $\eta$", size=16)
ax.set_ylabel(r"Time constant of synapses $\tau_s$", size=16)

for i in range(len(tau)):
    for j in range(len(rate)):
        text = ax.text(j, i, f"{d[i, j]}\n({100-d2[i, j]})",
                       ha="center", va="center", color="w")

plt.show()
