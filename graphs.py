"""Directed graph algorithm implementations."""


def creates_cycle(connections, test):
    """
    Returns true if the addition of the 'test' connection would create a cycle,
    assuming that no cycle already exists in the graph represented by 'connections'.
    """
    # funkcja ta szuka drogi od postSynTest do preSynTest
    # jeżeli ją znajdzie to pętla istnieje
    preSynTest, postSynTest = test
    # pętla sam do siebie (to już się wykonuje (liczy) w active net)
    if preSynTest == postSynTest:
        return True
    # nody, w których byłem, zaczynam od postSynTest
    visited = {postSynTest}
    while True:
        num_added = 0  # licznik otwartych nowych dróg
        # dla wszystkich połączeń
        for preSyn, postSyn in connections:
            # jezeli połączenie jest od noda do którego doszedłem
            # do noda, w którym jeszcze nie byłem (szukam tylko nowych dróg)
            if preSyn in visited and postSyn not in visited:
                # sprawdź czy jest to zamknięcie pętli, czyli czy trafiam do
                # do preSynTest
                if preSynTest == postSyn:  # jezeli tak to zwróc True pętli
                    return True
                # jeżeli nie, to dodaj postSyn do zbioru nodów do których mogę dojsc
                visited.add(postSyn)
                # i zwiększ licznik nowych nodów, do których mogę dojsc
                num_added += 1
        # jeżeli nie mogę dojsc już w żadne nowe miejsce
        # to zwróć False, bo nie znalazłem pętli a nowych dróg brak
        if num_added == 0:
            return False


def required_for_output(inputs, outputs, connections):
    """
    Collect the nodes whose state is required to compute the final network output(s).
    :param inputs: list of the input identifiers
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.
    NOTE: It is assumed that the input identifier set and the node identifier set are disjoint.
    By convention, the output node ids are always the same as the output index.

    Returns a set of identifiers of required nodes.
    """

    required = set(outputs)
    s = set(outputs)
    while 1:
        # Find nodes not in S whose output is consumed by a node in s.
        # wez nody presynaptyczne, ktore są w połączeniach, które
        # są do nodów w zbiorze s i nie są ze zbioru s
        t = set(a for (a, b) in connections if b in s and a not in s)

        if not t:
            break

        layer_nodes = set(x for x in t if x not in inputs)
        if not layer_nodes:
            break

        required = required.union(layer_nodes)
        s = s.union(t)

    return required


def feed_forward_layers(inputs, outputs, connections):
    """
    Dzieli neurony na warstwy w taki sposób, by wszystkie wejscia
    (OPROCZ wejscia od samego siebie) neuronu były okreslone
    w momencie jego obliczania. Dodatkowo neurony, które nie mają wpływu
    bezposredniego i posredniego na wyjscie sa pomijane).

    Collect the layers whose members can be evaluated in parallel in a feed-forward network.
    :param inputs: list of the network input nodes
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.

    Returns a list of layers, with each layer consisting of a set of node identifiers.
    Note that the returned layers do not contain nodes whose output is ultimately
    never used to compute the final network output.
    """

    required = required_for_output(inputs, outputs, connections)
    # lista warstw
    layers = []
    # to zbiór nodów które już zostały wykonane,
    # do niego dopisywane będą nody, które można juz obliczyć
    executedNodes = set(inputs)
    while 1:
        # Find candidate nodes for the next layer.  These nodes should connect
        # a node in ,,current" to a node not in ,,current".
        # wez kazdego noda docelowego - postynaptycznego w połączeniach
        # jezeli a jest w zbiorze wyjsciowych i b nie jest w zbiorze wyjsciowych
        candidates = set(postSyn for (preSyn, postSyn) in connections
                         if preSyn in executedNodes and postSyn not in executedNodes)
        # Keep only the used nodes whose entire input set is contained in executedNodes.
        layerNodes = set()
        # dla każdego kandydata na noda warstwy
        for node in candidates:
            # jezeli nod jest w wymaganych, czyli wpływa na wyjscie
            # i wszystkie połaczenia, które trafiają do
            # noda wychodzą (preSynaptyczne) wychodza z executedNodes,

            # czyli inaczej, jeżeli wszystkie presynaptyczne połączenia noda
            # należą do nodow z executedNodes

            # czyli jeszcze inaczej
            # sprawdza dla kazdego połaczenia w connections które trafia do noda
            # czy nod preSynaptyczny jest w executedNodes
            """Dodałem tutaj wyjątek, żeby dodawał połączenia do samego siebie,
            czyli zeby pomijał połączenia, w któryj node jest preSyn aptyczny"""
            if node in required and all(preSyn in executedNodes for (preSyn, postSyn)
                                        in connections if
                                        (postSyn == node and preSyn != node)):
                layerNodes.add(node)

        if not layerNodes:
            break
        # zapamiętaj kolejnosc wykonywania
        layers.append(layerNodes)
        # dodaj nody warstwy do juz przypisanych
        executedNodes = executedNodes.union(layerNodes)

    return layers


def dfs_recursive(node, visited, stack, connections, connToBreak):
    # Mark current node as visited and
    # adds to recursion stack
    visited[node] = True
    stack[node] = True
    # Recur for all postSyn nodes
    for preSyn, postSyn in [(preSyn, postSyn) for (preSyn, postSyn) in connections
                            if preSyn == node]:
        if visited[postSyn] is False:
            dfs_recursive(postSyn, visited, stack, connections, connToBreak)
        elif stack[postSyn] is True:
            connToBreak.add((preSyn, postSyn))
    stack[node] = False


def find_connections_to_break_in_cycles(inputs, outputsAndHidden, connections):
    """to jest algorytm DFS
    szuka wszysktich możliwych pętli i zwraca połaczenie które należy rozerwać

    params: nodes - nodes that is all hidden and outputs, whitout inputs,
    connections - direct connections"""
    connToBreak = set()
    visited = dict.fromkeys(inputs+outputsAndHidden, False)
    stack = dict.fromkeys(inputs+outputsAndHidden, False)
    # zacznij w każdym z nodów
    for startNode in inputs+outputsAndHidden:
        if visited[startNode] is False:
            dfs_recursive(startNode, visited, stack, connections, connToBreak)
    return connToBreak


def check_node_execution(node, conn, connToBrake, executedNodes):
    # znajdź wszystkie połączenia które są do noda
    # z nodów, które nie zostały wykonane
    connections_from_not_executed_nodes = [(preSyn, postSyn) for (preSyn, postSyn)
                                           in conn
                                           if postSyn == node and
                                           preSyn not in executedNodes]
    # jeżeli wszystkie te połączenia, to połączenia do rozerwania
    return all([conn in connToBrake for conn in connections_from_not_executed_nodes])


def find_free_nodes(nodes, conn):
    free = []
    for node in nodes:
        is_conn_to_node = next((preSyn for (preSyn, postSyn) in conn
                                if postSyn == node), False)
        if not is_conn_to_node:
            free.append(node)
    return free


def anyNet_layers(inputs, outputs, connections):
    """
    Dzieli neurony na warstwy w taki sposób, by wszystkie wejscia
    (OPROCZ wejscia od samego siebie) neuronu były okreslone
    w momencie jego obliczania. Dodatkowo neurony, które nie mają wpływu
    bezposredniego i posredniego na wyjscie sa pomijane).

    :param inputs: list of the network input nodes
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.

    Returns a list of layers, with each layer consisting of a set of node identifiers.
    Note that the returned layers do not contain nodes whose output is ultimately
    never used to compute the final network output.
    """

    required = required_for_output(inputs, outputs, connections)
    # weź tylko te połączenia, które są potrzebne
    # czyli te, które trafią do wykonywanych nodów lub wejsc
    # z wykonywanych nodów nie jest ujęte, ponieważ nod z którego
    # połączenie trafia do innego wykonywanego noda musi być w required
    # więc jego połączenia również zostaną przepisane
    conn = [(preSyn, postSyn) for (preSyn, postSyn) in connections
            if postSyn in required or postSyn in inputs]
    # szukaj pętli tworzonych przez nody które są wymagane,
    # znajdzie wszystie pętle oprócz simple circle, czyli sam do siebie
    connToBrake = find_connections_to_break_in_cycles(inputs, list(required), conn)
    # lista warstw, pierwsza warstwa to
    # nody, do których nie idą zadne połaczenia, ale sa one wymagane
    layers = []
    free_nodes = find_free_nodes(required, conn)
    if free_nodes:
        layers.append(find_free_nodes(required, conn))
    # to zbiór nodów które już zostały wykonane,
    # do niego dopisywane będą nody, które można juz obliczyć
    executedNodes = set(inputs + free_nodes)
    # wszystkie wymagane nody oprócz już wykonanych
    candidates = required - executedNodes
    # jeżeli jaki node jest required, a nigdy nie będzie wykonywany to petla
    # bedzie trwała wiecznosc, dlatego dodaje zabezpieczenie
    # największ możliwa opcja to wtedy, gdy kazdy neuron bedzie osobną warstwą
    maxIter = len(required)
    while candidates:
        # # dla każdego kandydata na noda warstwy sprawdź czy może być wykonany
        layerNodes = [n for n in candidates if
                      check_node_execution(n, conn, connToBrake, executedNodes)]
        # zapamiętaj kolejnosc wykonywania
        layers.append(layerNodes)
        # dodaj nody warstwy do juz przypisanych
        executedNodes = executedNodes.union(layerNodes)
        # oblicz nowych kandydatów, to samo co required - executedNodes
        candidates -= executedNodes
        maxIter -= 1
        if maxIter < 0:
            raise TimeoutError(
                'The loop creating the network layers takes too long, probably infinity.')
    return layers
