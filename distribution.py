import numpy as np
import math
from itertools import product

import settings as s
d = s.Distribution()


def fritz(parameter):  # Values for the 'Fritz' distribution ('triangle' configuration) (Calculation is described in
    dist = np.zeros((4, 4, 4))                                                                            # the Appendix
    d.qubits = np.zeros((4, 4, 4)).astype(str)

    for i, j, k, l, m, n in product('01', repeat=6):
        a = int(i + j, 2)
        b = int(k + l, 2)
        c = int(m + n, 2)
        d.qubits[a, b, c] = ''.join([str(a), str(b), str(c)])

        if m == j and n == l:
            if n == '0':
                if i == k:
                    dist[a, b, c] = (1 - parameter / np.sqrt(2)) / 16
                if i != k:
                    dist[a, b, c] = (1 + parameter / np.sqrt(2)) / 16
            if n == '1':
                if (m == '0' and i == k) or (m == '1' and i != k):
                    dist[a, b, c] = (1 - parameter / np.sqrt(2)) / 16
                if (m == '0' and i != k) or (m == '1' and i == k):
                    dist[a, b, c] = (1 + parameter / np.sqrt(2)) / 16

    d.qubits = d.qubits.flatten()
    return dist


def LLL(parameter):  # LLL-distribution with binary outputs, noise at the 4 points: 001, 010, 100, 110
    p0 = ((-1 + np.sqrt(5)) / 2)
    p1 = ((3 - np.sqrt(5)) / 2)

    d.qubits = np.zeros((2, 2, 2)).astype(str)
    dist = np.zeros((2, 2, 2))
    for a, b, c in product('01', repeat=3):
        d.qubits[int(a), int(b), int(c)] = ''.join([a, b, c])
        a, b, c = int(a), int(b), int(c)

        if c == (1 - a) * (1 - b):
            pa = p0 if (a == 0) else p1
            pb = p0 if (b == 0) else p1

            dist[a, b, c] = pa * pb

            if a == b:
                dist[a, b, c] += parameter
            else:
                dist[a, b, c] -= parameter

    d.qubits = d.qubits.flatten()
    return dist


def LLL_random_noise(parameter):  # LLL-distribution  with binary outputs
    p0 = ((3 - np.sqrt(5)) / 2)
    p1 = ((-1 + np.sqrt(5)) / 2)

    d.qubits = np.zeros((2, 2, 2)).astype(str)
    dist = np.zeros((2, 2, 2))
    for a, b, c in product('01', repeat=3):
        d.qubits[int(a), int(b), int(c)] = ''.join([a, b, c])
        a, b, c = int(a), int(b), int(c)

        if c == (1 - a) * (1 - b):
            pa = p0 if (a == 0) else p1
            pb = p0 if (b == 0) else p1

            dist[a, b, c] = pa * pb

    noise = np.random.rand(2, 2, 2)
    noise = noise / np.sum(noise)

    dist = (1 - parameter) * dist + parameter * noise
    d.qubits = d.qubits.flatten()
    return dist


def compile_model(distribution):  # Start the calculation of the distribution according to the wanted distribution

    if distribution == 'fritz':

        temp = np.zeros((len(d.para_range), 64))
        for i in range(d.para_range.shape[0]):  # Put the distributions with their different parameter together
            temp[i, :] = fritz(d.para_range[i]).flatten()

        return temp

    elif distribution == 'LLL':

        temp = np.zeros((len(d.para_range), 8))
        for i in range(d.para_range.shape[0]):
            temp[i, :] = LLL(d.para_range[i]).flatten()

        return temp

    elif distribution == 'LLL_random_noise':

        temp = np.zeros((len(d.para_range), 8))
        for i in range(d.para_range.shape[0]):
            temp[i, :] = LLL_random_noise(d.para_range[i]).flatten()

        return temp

    elif distribution == 'quantum calculated distribution':
        return quantum_distribution_computation()
    else:
        print('Distribution not defined, pls define it first in the file <<distribution.py>> or use an'
              ' existing distribution')
        return None


def new_distribution(distribution):  # Start: Initialise the parameters for the models

    d.save(distribution)
    d.name = distribution

    return compile_model(distribution)


def get_name():
    return d.name


def get_para_range():
    return d.para_range


def set_connection_between_states(c_b_s):
    d.connection_between_states = c_b_s


def set_target(t):
    d.target = t


def print_distribution(distribution):  # Print the distribution which the corresponding states

    if d.name == 'quantum calculated distribution':
        distribution = distribution.flatten()
        prt = np.array(distribution, dtype=str)
        for i in range(len(distribution)):
            prt[i] = '{}: {} '.format(d.qubits[i], round(distribution[i], 9))

        for i in range(int(len(prt) / 4)):
            print(prt[4 * i:4 * (i + 1)])
    else:
        for i in range(distribution.shape[0]):
            print('\nParameter: ', round(d.para_range[i], 6))
            prt = np.array(distribution[i], dtype=str)
            for j in range(distribution.shape[1]):
                a = round(distribution[i, j], 7)
                prt[j] = '{}: {} '.format(d.qubits[j], a)
            for j in range(int(len(prt) / 4)):
                print(prt[4 * j:4 * (j + 1)])


def quantum_distribution_computation():  # Calculate the quantum distribution
    psi = defining_psi()
    alice_state, bob_state, charlie_state = defining_manual_states()
    print('\nalice:\n', alice_state, '\nbob:\n', bob_state, '\ncharlie:\n', charlie_state)

    dist = np.zeros((4, 4, 4))
    d.qubits = np.zeros((4, 4, 4)).astype(str)
    for a, b, c in product('0123', repeat=3):
        a, b, c = int(a), int(b), int(c)
        d.qubits[a, b, c] = ''.join([str(a), str(b), str(c)])

        if np.any(alice_state[a, 0:3] != charlie_state[c, 0:3]) or np.any(bob_state[b, 0:3] != charlie_state[c, 3:6]):
            continue

        dist[a, b, c] = single_quantum_computation(alice_state[a, :], bob_state[b, :], psi)

    d.qubits = d.qubits.flatten()
    print(dist.flatten())
    return dist


def defining_psi(i=1/math.sqrt(2), j=0, k=0, q=1/math.sqrt(2)):  # Defining the 'psi' vector for further computation
    psi = np.array([i, j, k, q])
    psi = psi / math.sqrt(np.sum(psi**2, axis=0))  # Normalise the vector
    return psi


def defining_measurement_operator(output, which_matrix, operator='N', other_matrix='N'):
    # Defining the measurement operator according the pauli matrices
    if which_matrix == 'X':
        which_matrix = np.array([[0, 1], [1, 0]])
    elif which_matrix == 'Y':
        which_matrix = np.array([[0, -1j], [1j, 0]])
    elif which_matrix == 'Z':
        which_matrix = np.array([[1, 0], [0, -1]])
    else:
        print("Not a pauli matrix defined, please use one of the three pauli matrices ('X', 'Y' or 'Z')")
        return None

    if operator != 'N':
        if other_matrix == 'X':
            other_matrix = np.array([[0, 1], [1, 0]])
        elif other_matrix == 'Y':
            other_matrix = np.array([[0, -1j], [1j, 0]])
        elif other_matrix == 'Z':
            other_matrix = np.array([[1, 0], [0, -1]])
        else:
            print("Not a pauli matrix defined, please use one of the three pauli matrices ('X', 'Y' or 'Z')")
            return None

        if operator == '+':
            which_matrix = (which_matrix + other_matrix) / math.sqrt(2)
        elif operator == '-':
            which_matrix = (which_matrix - other_matrix) / math.sqrt(2)
        else:
            print("Not a valuable operator: ", operator, ", please use '+' or '-' as an operator or define it first")
            return None

    eigenvalue, eigenvector = np.linalg.eig(which_matrix)
    return np.outer(eigenvector[:, output - d.swap_eigenvector], eigenvector[:, output - d.swap_eigenvector])


def defining_states(s1='X', s2='Z'):  # An automatic definition of states which stats Alice, Bob and Charlie have
    a = np.empty((4, 4), dtype=str)
    b = np.empty((4, 4), dtype=str)
    c = np.empty((4, 6), dtype=str)

    c[0] = [s1, None, None, s1, '+', s2]
    c[1] = [s1, None, None, s1, '-', s2]
    c[2] = [s2, None, None, s1, '+', s2]
    c[3] = [s2, None, None, s1, '-', s2]

    a[:, 0] = c[:, 0]
    b[:, 0] = c[:, 1]

    st = [a, b]
    for i in range(2):
        x = st[i]
        if x[0, :] == x[1, :]:
            x[:, 4] = ['0', '1', '0', '1']
        else:
            x[:, 4] = ['0', '0', '1', '1']

    return a, b, c


def defining_manual_states():  # Here you can manually define specific states for Alice, Bob and Charlie

    # Be aware that you follow the rules mentioned in the paper """How to reproduce a quantum distribution""". Otherwise
    # it can't be reproduced by the program. If you're uncertain, then use 'defining_states'

    a = np.empty((4, 4), dtype=str)
    b = np.empty((4, 4), dtype=str)
    c = np.empty((4, 6), dtype=str)

    a[0] = ['Z', '+', 'Y', '0']
    a[1] = ['Z', '-', 'Y',  '1']
    a[2] = ['Z', '-', 'Y',  '0']
    a[3] = ['Z', '+', 'Y',  '1']

    b[0] = ['Z', 'N', 'N', '1']
    b[1] = ['Y', 'N', 'N', '0']
    b[2] = ['Z', 'N', 'N', '0']
    b[3] = ['Y', 'N', 'N', '1']

    c[0] = ['Z', '-', 'Y',  'Z', 'N', 'N']
    c[1] = ['Z', '+', 'Y',  'Y', 'N', 'N']
    c[2] = ['Z', '-', 'Y',  'Y', 'N', 'N']
    c[3] = ['Z', '+', 'Y',  'Z', 'N', 'N']

    return a, b, c


def create_starting_point(state, modus):  # Defining a first possible combination of states
    s0 = state[0]
    s1 = state[1]
    s2 = state[2]

    a = np.empty((4, 4), dtype=str)
    b = np.empty((4, 4), dtype=str)
    c = np.empty((4, 6), dtype=str)

    cbs = d.connection_between_states
    for i in range(cbs.shape[0]):
        a_connection = [cbs[i, 0], cbs[i, 2], cbs[i, 4], cbs[i, 6]]
        b_connection = [cbs[i, 1], cbs[i, 3], cbs[i, 5], cbs[i, 7]]

        if modus == 0:
            if i == 0:
                c[i] = [s0, 'N', 'N', s0, '+', s1]
            else:
                c[i, 0:3] = a[d.connection_between_states[i, 0], 0:3]
                c[i, 3:6] = b[d.connection_between_states[i, 1], 0:3]
                if np.all(c[i, 0:3] == ''):
                    c[i, 0:3] = [s1, 'N', 'N']
                if np.all(c[i, 3:6] == ''):
                    c[i, 3:6] = [s0, '-', s1]

        elif modus == 1:
            if i == 0:
                c[i] = [s0, '+', s1, s0, '+', s2]
            else:
                c[i, 0:3] = a[d.connection_between_states[i, 0], 0:3]
                c[i, 3:6] = b[d.connection_between_states[i, 1], 0:3]
                if np.all(c[i, 0:3] == ''):
                    c[i, 0:3] = [s0, '-', s1]
                if np.all(c[i, 3:6] == ''):
                    c[i, 3:6] = [s0, '-', s2]

        elif modus == 2:
            if i == 0:
                c[i] = [s0, 'N', 'N', s0, 'N', 'N']
            else:
                c[i, 0:3] = a[d.connection_between_states[i, 0], 0:3]
                c[i, 3:6] = b[d.connection_between_states[i, 1], 0:3]
                if np.all(c[i, 0:3] == ''):
                    c[i, 0:3] = [s1, 'N', 'N']
                if np.all(c[i, 3:6] == ''):
                    c[i, 3:6] = [s2, 'N', 'N']

        else:
            print('Error, wrong modus choosen, pls choose modus = {0, 1, 2} or define a new one')
            return None

        a[min(a_connection), 0:3] = c[i, 0:3]
        a[max(a_connection), 0:3] = c[i, 0:3]
        b[min(b_connection), 0:3] = c[i, 3:6]
        b[max(b_connection), 0:3] = c[i, 3:6]

    st = [a, b]
    for i in range(2):
        x = st[i]
        if np.all(x[0, :] == x[1, :]):
            x[:, 3] = ['0', '1', '0', '1']
        else:
            x[:, 3] = ['0', '0', '1', '1']

    return a, b, c


def single_quantum_computation(a, b, psi=defining_psi()):  # Calculate p(x) =〈Ψ|M|Ψ〉=tr(|Ψ〉〈Ψ|M)
    ma0, mao, ma1 = a[0], a[1], a[2]
    oa = int(a[3])

    mb0, mbo, mb1 = b[0], b[1], b[2]
    ob = int(b[3])

    ma = defining_measurement_operator(oa, ma0, mao, ma1)
    mb = defining_measurement_operator(ob, mb0, mbo, mb1)

    measurement_operator = np.kron(ma, mb)
    return abs(np.trace(np.matmul(np.outer(psi, psi), measurement_operator))) / 4


def control_connections(a, b, c):

    for i in range(c.shape[0]):
        for j in range(int(d.connection_between_states.shape[1]/2)):

            if a[d.connection_between_states[i, 2*j], 0] != c[i, 0] and \
                    np.all(a[d.connection_between_states[i, 2*j]] != ''):
                print('False connection:a ', a[d.connection_between_states[i, 2*j], 0], ' --- ', c[i, 0])
                return False

            if np.all(b[d.connection_between_states[i, 2*j + 1], 0:3] != c[i, 3:6]) and \
                    np.all(b[d.connection_between_states[i, 2*j]] != ''):
                print('False connection:b ', b[d.connection_between_states[i, 2*j + 1], 0:3], ' --- ',
                      c[i, 3], c[i, 4], c[i, 5])
                return False

    return True


def swap(state, x, y):  # Swap two states of a state vector
    state[x], state[y] = state[y].copy(), state[x].copy()
    return state


def control_already_set_combination_of_states(alice, bob, a0, b0, c0):  # Control if the new states work also for the
    for a, b, c in product('0123', repeat=3):                                      # previous points of the distribution
        a, b, c = int(a), int(b), int(c)
        if d.target[a, b, c] == 0:
            continue

        res = single_quantum_computation(alice[a], bob[b])
        if round(res, 10) != round(d.target[a, b, c], 10):
            return False

        if a == a0 and b == b0 and c == c0:
            return True


def try_different_measurement_outputs(alice, bob, a, b, c):  # Try other possible combination of states according the
    alice0, bob0 = alice.copy(), bob.copy()                                                                      # rules

    for i in range(alice.shape[0]):
        if np.all(alice[a, 0:3] == alice[i, 0:3]) and a != i:
            alice = swap(alice, a, i)
    if control_already_set_combination_of_states(alice, bob, a, b, c):
        return alice, bob, True
    alice = alice0.copy()

    for i in range(bob.shape[0]):
        if np.all(bob[b, 0:3] == bob[i, 0:3]) and b != i:
            bob = swap(bob, b, i)
    if control_already_set_combination_of_states(alice, bob, a, b, c):
        return alice, bob, True

    for i in range(alice.shape[0]):
        if np.all(alice[a, 0:3] == alice[i, 0:3]) and a != i:
            alice = swap(alice, a, i)
    if control_already_set_combination_of_states(alice, bob, a, b, c):
        return alice, bob, True
    else:
        return alice0, bob0, False


def try_different_combination_of_states(alice, bob, charlie, a, b, c):  # Control if the states lead to the correct
    res = single_quantum_computation(alice[a], bob[b])              # distribution, otherwise swap measurement operators
    if round(res, 10) != round(d.target[a, b, c], 10):
        # print('Res: {}, target: {}, abc: {}, {}, {}.'.format(res, d.target[a, b, c], a, b, c))
        # print("alice:\n", alice, "\nbob:\n", bob, "\ncharlie:\n", charlie)
        alice, bob, success = try_different_measurement_outputs(alice, bob, a, b, c)
        if success:
            print("Success: alice:\n", alice, "\nbob:\n", bob, "\ncharlie:\n", charlie)
        else:
            print("No success: alice:\n", alice, "\nbob:\n", bob, "\ncharlie:\n", charlie)

    return alice, bob, charlie
