from geneal.genetic_algorithms import ContinuousGenAlgSolver, BinaryGenAlgSolver
from collections import Counter
import random
import numpy as np

random.seed(10)

def convert_tree_to_prufer(tree: list, n):
    temp_tree = tree.copy()
    prufer = []
    while len(temp_tree) != 1:
        for i in range(1, n+1):
            to_delete = []
            for t in temp_tree:
                if t[0] == i or t[1] == i:
                    to_delete.append(t)
            if len(to_delete) == 1:
                if to_delete[0][0] == i:
                    prufer.append(to_delete[0][1])
                elif to_delete[0][1] == i:
                    prufer.append(to_delete[0][0])
                temp_tree.remove(to_delete[0])
                break

    return prufer


def convert_prufer_to_tree(supplies:dict, demands:dict, prufer: list, n):
    tree = []
    temp_prufer = list(prufer).copy()
    not_prufer = list(set([i for i in range(1, n+1)]) - set(temp_prufer))
    temp_supplies = supplies.copy()
    temp_demands = demands.copy()
    while temp_prufer:
        i = not_prufer[0]
        j = temp_prufer[0]
        index_to_delete = 0
        if (i in temp_supplies and j in temp_supplies) or (i in temp_demands and j in temp_demands):
            for index, k in enumerate(temp_prufer):
                if not((i in temp_supplies and k in temp_supplies) or (i in temp_demands and k in temp_demands)):
                    j = k
                    index_to_delete = index
                    break
        print(not_prufer)
        print(temp_prufer)
        not_prufer.pop(0)
        temp_prufer.pop(index_to_delete)
        if j not in temp_prufer:
            not_prufer.append(j)
            not_prufer.sort()
        print(temp_supplies)
        print(temp_demands)
        print(i)
        print(j)
        print("++++++++++++++++++++++++++++++++++++++++")
        x = min(temp_supplies[min(i, j)], temp_demands[max(j, i)])
        temp_supplies[min(i, j)] -= x
        temp_demands[max(j, i)] -= x
        edge = (min(i, j), max(j, i), x)
        tree.append(edge)

    print(not_prufer)
    print(prufer)
    x = min(temp_supplies[not_prufer[0]], temp_demands[not_prufer[1]])
    temp_supplies[not_prufer[0]] -= x
    temp_demands[not_prufer[1]] -= x

    edge = (not_prufer[0], not_prufer[1], x)
    tree.append(edge)
    if all(value == 0 for value in temp_demands.values()) and all(value == 0 for value in temp_supplies.values()):
        return tree

    r = 0
    s = 0
    for key, value in temp_supplies.items():
        if value != 0:
            r = key

    for key, value in temp_demands.items():
        if value != 0:
            s = key

    tree.append((r, s, temp_supplies[r]))

    to_remove = []
    for i in range(len(tree)):
        for j in range(i+1, len(tree)):
            if tree[i][0] == tree[j][1] and tree[i][1] == tree[j][0]:
                if tree[i][2] == 0:
                    to_remove.append(tree[i])
                elif tree[j][2] == 0:
                    to_remove.append(tree[j])

    for elem in to_remove:
        tree.remove(elem)

    return tree


class TransportationProblemSolver(ContinuousGenAlgSolver, BinaryGenAlgSolver):
    def __init__(self, *args, **kwargs):
        self.demands = kwargs.pop('demands')
        self.supplies = kwargs.pop('supplies')
        self.inversion_mutation_rate = kwargs.pop('inversion_mutation_rate')
        self.displacement_mutation_rate = kwargs.pop('displacement_mutation_rate')
        self.cost = kwargs.pop('cost')
        BinaryGenAlgSolver.__init__(self, *args, **kwargs)
        ContinuousGenAlgSolver.__init__(self, *args, **kwargs)

    def fitness_function(self, chromosome):
        tree = convert_prufer_to_tree(self.supplies, self.demands, chromosome, self.n_genes)
        sum = 0

        print(chromosome)
        print(tree)
        print(self.supplies)
        print(self.demands)
        print(self.cost)
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
        for branch in tree:
            print(branch)
            c = self.cost[branch[0] - 1][branch[1] - len(self.cost) - 1]
            sum += c * branch[2]
        print("=====================================================")
        return sum

    def initialize_population(self):
        population = []
        while len(population) < self.pop_size:
            print(len(population))
            prufer = [random.randint(1, self.n_genes + 1) for i in range(self.n_genes - 2)]
            not_prufer = list(set([i for i in range(1, self.n_genes + 1)]) - set(prufer))
            c = Counter(prufer)
            so = 0
            sd = 0
            nso = 0
            nsd = 0
            for k, c in c.items():
                if k in self.supplies.keys():
                    so += c + 1
                elif k in self.demands.keys():
                    sd += c + 1
            for k in not_prufer:
                if k in self.supplies.keys():
                    nso += 1
                elif k in self.demands.keys():
                    nsd += 1
            if so + nso != sd + nsd:
                continue
            try:
                convert_prufer_to_tree(self.supplies, self.demands, prufer, self.n_genes)
            except Exception:
                pass
            else:
                population.append(prufer)

        return np.array(population)

    def create_offspring(
        self, first_parent, sec_parent, crossover_pt, offspring_number
    ):
        if offspring_number == 'first':
            fa = first_parent[0:crossover_pt[0]]
            fb = sec_parent[crossover_pt[0]:]
            return np.array(list(fa) + list(fb))
        return np.array(list(sec_parent[0:crossover_pt[0]]) + list(first_parent[crossover_pt[0]:]))

    def mutate_population(self, population, n_mutations):
        def reverse_sublist(lst, start, end):
            lst[start:end] = lst[start:end][::-1]
            return lst

        def subshift(L, start, end, insert_at):
            temp = L[start:end]
            L = L[:start] + L[end:]
            return L[:insert_at] + temp + L[insert_at:]

        print(population)
        pop_copy = population.copy()
        for i in range(len(population)):
            for j in range(n_mutations):
                if random.randint(0, self.displacement_mutation_rate + self.inversion_mutation_rate) < self.inversion_mutation_rate:
                    start_position = random.randint(0, len(pop_copy[i]))
                    stop_position = random.randint(start_position, len(pop_copy[i]))
                    pop_copy[i] = np.array(reverse_sublist(list(pop_copy[i]), start_position, stop_position))
                else:
                    start_position = random.randint(0, len(pop_copy[i]))
                    stop_position = random.randint(start_position, len(pop_copy[i]))
                    l = random.randint(1, len(pop_copy[i] - (stop_position - start_position)))
                    pop_copy[i] = np.array(subshift(list(pop_copy[i]), start_position, stop_position, l))

        return pop_copy

supplies = {
    1: 8,
    2: 19,
    3: 17,
    4: 10
}

demands = {
    5: 11,
    6: 9,
    7: 5,
    8: 14,
    9: 15
}

cost = [[4, 4, 3, 4, 8], [5, 8, 9, 10, 15], [6, 2, 5, 1, 12], [3, 5, 6, 9, 9]]

print(convert_prufer_to_tree(supplies.copy(), demands.copy(), [5, 9, 2, 3, 2, 8, 3], 9))
print("+++++++++++++++++++++++++++++")
print(convert_tree_to_prufer(convert_prufer_to_tree(supplies.copy(), demands.copy(), [5, 9, 2, 3, 2, 8, 3], 9),9))
print(convert_tree_to_prufer([(1, 5, 8), (4, 9, 10), (2, 5, 3), (3, 6, 9), (2, 7, 5), (2, 8, 11), (3, 8, 3), (3, 9, 5)], 9))

solver = TransportationProblemSolver(supplies=supplies, demands=demands, cost=cost, mutation_rate=0.1, selection_rate=0.6, selection_strategy='tournament', pop_size=100, max_gen=200, n_genes=9, inversion_mutation_rate=50, displacement_mutation_rate=50, n_crossover_points=1)
solver.solve()