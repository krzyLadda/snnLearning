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

data = pd.read_excel("plot ile zbieglo.ods", engine="odf")
rate = [x for x in data.columns[1:]]
tau = data['tau'].tolist()

values = {}
for row, t in zip(data.iloc, tau):
    foo = row[1:].tolist()
    values[t] = [100 - x for x in foo]

plt.figure(1)
markers = ["o", "v", "x", "D", "+", "s", "*", "^"]
ax = plt.gca()

ax.set_xscale('log')
for i, t in enumerate(tau):
    l = r'$\tau_s = '+str(t)+r'$'
    plt.plot(rate, values[t], marker=markers[i], label=l)
    for i, v in enumerate(values[t]):
        if np.isnan(v):
            continue
        #ax.text(rate[i], v, v, size=12)

ax.legend(ncol=int(len(tau)/2), loc='lower left', bbox_to_anchor=(0.0, 1.0), fancybox=True)

foo = np.array(list(values.values()))
plt.ylim([0.95*np.nanmin(foo), 1.05*np.nanmax(foo)])
plt.xlim([0.95*rate[0], 1.05*rate[-1]])

ax.set_xticks(rate)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
yticks = list(range(0, 21, 5)) + list(range(20, 101, 10))
ax.set_yticks(yticks)

plt.grid(linewidth=0.4)
plt.xticks(rotation=45)

plt.xlabel(r"Learning rate $\eta$")
plt.ylabel("Number of did not converge calls")
plt.show()
