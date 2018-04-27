def make_arr(shape, p):
    f = lambda x: -1 if x >= p else 1
    f = np.vectorize(f)
    return f(np.random.rand(*shape))

def flip_spins(arr, flip_indices):
    new_arr = arr.copy()
    for i in flip_indices:
        new_arr[i] = -arr[i]
    return new_arr

def get_neighbours(shape, index_tuple):
    neighbours = []
    if index_tuple[0] > 0:
        neighbours.append((index_tuple[0] - 1, index_tuple[1]))
    if index_tuple[1] > 0:
        neighbours.append((index_tuple[0], index_tuple[1] - 1))
    if index_tuple[0] < shape[0] - 1:
        neighbours.append((index_tuple[0] + 1, index_tuple[1]))
    if index_tuple[1] < shape[1] - 1:
        neighbours.append((index_tuple[0], index_tuple[1] + 1))
    return neighbours

def magnetization(arr, **kwargs):
    return sum(arr.flatten())

def magnetization_2(arr, **kwargs):
    return abs(sum(arr.flatten()))

def P(arr, vertical=True, horizontal=True):
    N, l_dict = cluster_labeled_arr(arr)
    left = {l_dict[k] if k in l_dict else k for k in set(N[:, 0])}
    right = {l_dict[k] if k in l_dict else k for k in set(N[:, -1])}
    top = {l_dict[k] if k in l_dict else k for k in set(N[0])}
    bottom = {l_dict[k] if k in l_dict else k for k in set(N[-1])}
    perc = False
    if vertical and horizontal:
        perc = any(set.intersection(left, right, top, bottom))
    elif vertical:
        perc = any(set.intersection(top, bottom))
    else:
        perc = any(set.intersection(left, right))
    return int(perc)
    
    
def cluster_labeled_arr(arr):
    counter = 1
    N = np.zeros(arr.shape)
    l_dict = {}
    for i, row in enumerate(arr):
        for j, elem in enumerate(row):
            if elem == 1:     
                neighbours = get_neighbours(arr.shape, (i, j))
                neighbour_labels = []
                for n in neighbours:
                    if N[n] != 0 and N[n] not in neighbour_labels:
                        neighbour_labels.append(N[n])
                if not neighbour_labels:
                    N[i, j] = counter
                    counter += 1
                else:
                    neighbour_labels.sort()
                    min_label = neighbour_labels.pop(0)
                    N[i, j] = min_label
                for n_l in neighbour_labels:
                    l_dict[n_l] = min_label

    d_counter = 0
    while d_counter < len(l_dict):
        for k, v in l_dict.items():
            if v in l_dict.keys():
                d_counter = 0
                l_dict[k] = l_dict[v]
            d_counter += 1
    return N, l_dict
                

def cluster_sizes(arr):
    arr = m
    N, l_dict = cluster_labeled_arr(arr)
    cl_labels = set(N.flatten())
    cl_labels.remove(0)
    for k in l_dict.keys():
        cl_labels.remove(k)
    cl_sizes = []
    for label in cl_labels:
        val = list(N.flatten()).count(label)
        for k, v in l_dict.items():
            if label == v:
                val += list(N.flatten()).count(k)
        cl_sizes.append(val)
    return cl_sizes


def num_of_clusters(arr, **kwargs):
    N, l_dict = cluster_labeled_arr(arr)
    s = set(N.flatten())
    s.remove(0)
    return len(s) - len(l_dict)


def ave_cluster_size(arr, **kwargs):
    cl_s = cluster_sizes(arr)
    return sum(cl_s) / len(cl_s)


def energy(J, arr, **kwargs):
    E = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if i < arr.shape[0] - 1:
                E += -J if arr[i, j] == arr[i + 1, j] else J
            if j < arr.shape[1] - 1:
                E += -J if arr[i, j] == arr[i, j + 1] else J
    return E               

def exact_algorithm(shape, T, J, thing_to_measure, **kwargs):
    combinations = int('1' * shape[0] * shape[1], 2) + 1
    expectation_val = 0
    partition_fn = 0
    for i in range(combinations):
        str_vec_repr = bin(i)[2:].zfill(shape[0] * shape[1])
        arr = np.array([int(n) for n in str_vec_repr]).reshape(shape[0], shape[1])
        arr = arr * 2 - 1
        prob = np.exp(-1/T * energy(J, arr))
        expectation_val += thing_to_measure(arr, **kwargs) * prob
        partition_fn += prob
    return expectation_val / partition_fn

def metropolis_algorithm(shape, T, J, steps, thing_to_measure, **kwargs):
    old_arr = make_arr(shape, 0.5)
    val = 0
    for i in range(steps):
        flip_ind_i = np.random.randint(0, old_arr.shape[0])
        flip_ind_j = np.random.randint(0, old_arr.shape[1])
        new_arr = flip_spins(old_arr, [(flip_ind_i, flip_ind_j)])
        prob = np.exp(-1/T * (energy(J, new_arr) - energy(J, old_arr)))
        if np.random.rand(1) < prob:
            old_arr = new_arr
        val += thing_to_measure(old_arr, **kwargs) / steps
    return val
#     return thing_to_measure(old_arr, **kwargs)


def wolff_algorithm(shape, T, J, steps, thing_to_measure, **kwargs):
    old_arr = make_arr(shape, 0.5)
    prob = 1 - np.exp(-2 / T * J)
    val = 0
    for i in range(steps):
        k = np.random.randint(0, shape[0])
        j = np.random.randint(0, shape[1])
        cluster_indices = []
        rejected_indices = []
        new_indices = [(k, j)]
        while True:
            try:
                test  = new_indices.pop()
            except IndexError:
                break
            cluster_indices.append(test)
            neighbours = get_neighbours(shape, test)
            for neighbour in neighbours:
                if neighbour not in cluster_indices + rejected_indices:
                    if old_arr[test] == old_arr[neighbour] and np.random.rand() < prob:
                        new_indices.append(neighbour)
                    else:
                        rejected_indices.append(neighbour)
        val += thing_to_measure(old_arr, **kwargs) / steps
        old_arr = flip_spins(old_arr, cluster_indices)
    return val
#     return thing_to_measure(old_arr, **kwargs)