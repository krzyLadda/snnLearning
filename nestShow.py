import nest
import matplotlib.pyplot as plt
import numpy as np


def showNeurons(IDs, net, whatShow):
    # pokazuje neurony z sieci
    # dla każdego neuronu
    for i, n_id in enumerate(IDs):
        try:  # zobacz czy jest w warstwie ukrytej
            idx = net.Vneurons.index(n_id)
            # jeżeli neuron jest w warstwie ukrytej
            n_multimeter = nest.GetStatus(net.Vmultimeters)[idx]
            n_spikedetector = nest.GetStatus(net.Vspikedetectors)[idx]
        except ValueError:  # jeżeli nie
            try:  # zobacz czy jest w warstwie wyjsciowej
                idx = net.Wneurons.index(n_id)
                # jeżeli neuron jest w warstwie wyjściowej
                n_multimeter = nest.GetStatus(net.Wmultimeters)[idx]
                n_spikedetector = nest.GetStatus(net.Wspikedetectors)[idx]
            except ValueError as e:  # jeżeli nie ma go ani tu ani tu
                print("Error - neuron not found")
                print(e)
                raise  # zatrzymaj program z blędem
        showNeuron(n_multimeter, n_spikedetector, n_id, whatShow, i)


def showNeuron(multimeter, spikedetector, neuronID, whatShow, nr_fig=1):
    m_events = multimeter['events']
    s_events = spikedetector['events']

    hmSubplots = len(whatShow)
    fig = plt.figure(nr_fig)
    for i in range(1, hmSubplots+1):
        if i == 1:
            ax = fig.add_subplot(hmSubplots, 1, i)
            ax.set_title(f'Neuron {neuronID}')
        else:
            ax = fig.add_subplot(hmSubplots, 1, i, sharex=ax)

        whatS = whatShow[i-1]
        if (whatS in m_events or 'multimeter.' in whatS) and 'spikedetector.' not in whatS:
            if 'multimeter' in whatS:
                whatS = whatS.replace("multimeter.", "")
            source = m_events
            time = m_events['times']
            lineStyle = '-'
        elif whatS in s_events or 'spikedetector.' in whatS:
            if 'spikedetector.' in whatS:
                whatS = whatS.replace('spikedetector.', "")
            source = s_events
            time = s_events['times']
            lineStyle = '.'
        else:
            print("Szukanego przebiegu nie ma w zasobach")
            raise
        ax.plot(time, source[whatS], lineStyle)
        if whatShow[i-1] == 'spikedetector.senders':
            ax = fig.gca()
            for x, y in zip(source['times'], source['senders']):
                ax.text(x, y+0.05, "%f" % x, ha='center')
        ax.set_ylabel(whatS)
    ax.set_xlabel(r'$t [ms]$')
    plt.show()


def showOneNeuron(multimeter, spikedetector, neuronID, whatShow, nr_fig=1):
    plt.rcParams['text.usetex'] = True
    sd = nest.GetStatus(spikedetector, keys="events")[neuronID]
    m = nest.GetStatus(multimeter)[neuronID]['events'][neuronID]
    showNeuron(m, sd, neuronID, whatShow, nr_fig)
