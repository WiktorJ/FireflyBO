from abc import ABCMeta
import random
import numpy
import csv


class Generator(metaclass=ABCMeta):
    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def c(self):
        return self._c

    @property
    def alpha(self):
        return self._alpha

    @property
    def gamma(self):
        return self._gamma

    @property
    def beta(self):
        return self._beta

    @property
    def zeta(self):
        return self._zeta

    @property
    def l(self):
        return self._l

    @property
    def pmax(self):
        return self._pmax

    @property
    def pmin(self):
        return self._pmin

    @a.setter
    def a(self, a):
        self._a = a

    @b.setter
    def b(self, b):
        self._b = b

    @c.setter
    def c(self, c):
        self._c = c

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma

    @beta.setter
    def beta(self, beta):
        self._beta = beta

    @zeta.setter
    def zeta(self, zeta):
        self._zeta = zeta

    @l.setter
    def l(self, l):
        self._l = l

    @pmax.setter
    def pmax(self, pmax):
        self._pmax = pmax

    @pmin.setter
    def pmin(self, pmin):
        self._pmin = pmin

    def args(self):
        return [self.a, self.b, self.c, self.alpha, self.beta, self.gamma, self.zeta, self.l, self.pmin, self.pmax]

    def __str__(self):
        return "{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}" \
            .format(self.a, self.b, self.c, self.alpha, self.beta, self.gamma, self.zeta, self.l, self.pmin, self.pmax)


class RandomizedGenerator(Generator):
    def __init__(self):
        self._a = random.randint(2, 6) * 20
        self._b = random.randint(10, 20) * 10
        self._c = random.randint(2, 4) * 5
        self._alpha = (random.random() * 4.) + 2.5
        self._beta = -((random.random() * 3.5) + 3)
        self._gamma = (random.random() * 4.) + 3
        self._zeta = (random.random() * 9.99e-4) + 1.e-6
        self._l = (random.random() * 6.) + 2.
        self._pmin = random.randint(4, 6) / 100.
        self._pmax = random.randint(4, 12) / 10.


class ProvidedGenerator(Generator):
    def __init__(self, a, b, c, alpha, beta, gamma, zeta, l, pmin, pmax):
        self._a = a
        self._b = b
        self._c = c
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._zeta = zeta
        self._l = l
        self._pmin = pmin
        self._pmax = pmax


# returns tuple: (vector of generators, matrix of losses, demand, max total)
def generators_from_file(coef_path="resources/data1.csv", loss_path="resources/loss1.csv"):
    f = open(coef_path, 'r')
    b = open(loss_path, 'r')
    v = []
    B = []
    try:
        reader = csv.reader(f)
        loss_reader = csv.reader(b)
        D = next(reader)
        for row in reader:
            v.append(ProvidedGenerator(
                float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]),
                float(row[7]), float(row[8]), float(row[9])))
        for row in loss_reader:
            tmp = []
            for cell in row:
                tmp.append(float(cell))
            B.append(tmp)
    finally:
        f.close()

    ret = v, B, float(D[0]), calc_max_value(v)
    print_case(ret)
    return ret


# returns tuple: (vector of generators, matrix of losses, demand, max total)
def generate_randomized_generators(n, loss_empty=False):
    v = [RandomizedGenerator() for i in range(n)]
    min_sum = 0
    max_sum = 0
    for g in v:
        min_sum += g.pmin
        max_sum += g.pmax

    D = (random.random() * ((max_sum - min_sum) / 4.)) + min_sum + ((max_sum - min_sum) / 10.)
    if loss_empty:
        ret = v, numpy.zeros((n, n)), D, calc_max_value(v)
    else:
        ret = v, get_randomized_loss_matrix(n), D, calc_max_value(v)

    print_case(ret)
    return ret


def calc_max_value(v):
    max = 0
    for g in v:
        max += g.pmax
    return max + 1


def get_randomized_loss_matrix(n):
    loss_matrix = numpy.zeros((n, n))
    for i in range(0, n):
        for j in range(i, n):
            loss_matrix[i, j], loss_matrix[j, i] = random_loss_value()
            if i != j and random.random() < 0.54:
                loss_matrix[i, j] = - loss_matrix[i, j]
                loss_matrix[j, i] = - loss_matrix[j, i]
    return loss_matrix

    # numpy.savetxt("loss5.csv", loss_matrix, delimiter=",", fmt='%1.8f')


# 0-0.0475 1
# 0.0476-0.19 2
# 0.191 – 0.476 3
# 0.0477 – 0.9035 4
# 0.9036 – 1 5
def random_loss_value():
    v = random.random()
    if v <= 0.0475:
        val = round(random.uniform(0.01, 0.1), 4)
        return val, get_d_val(val)
    elif v <= 0.19:
        val = round(random.uniform(0.001, 0.01), 5)
        return val, get_d_val(val)
    elif v <= 0.476:
        val = round(random.uniform(0.0001, 0.001), 6)
        return val, get_d_val(val)
    elif v <= 0.9035:
        val = round(random.uniform(0.00001, 0.0001), 7)
        return val, val
    else:
        val = round(random.uniform(0.000001, 0.00001), 8)
        return val, val


def get_d_val(val):
    return val / 100 if random.randint(0, 2) == 0 else val


def print_case(tuple):
    generators = tuple[0]

    print("===================================== INPUT ====================================")
    print("                ", end="")
    for i in range(0, len(generators)):
        print("   PG%02d    " % i, end="")
    print("")

    print("cost     a      ", end="")
    for g in generators:
        print("%10f " % g.a, end="")
    print("")

    print("         b      ", end="")
    for g in generators:
        print("%10f " % g.b, end="")
    print("")

    print("         c      ", end="")
    for g in generators:
        print("%10f " % g.c, end="")
    print("")

    print("--------------------------------------------------------------------------------")

    print("emmision alfa   ", end="")
    for g in generators:
        print("%10f " % g.alpha, end="")
    print("")

    print("         beta   ", end="")
    for g in generators:
        print("%10f " % g.beta, end="")
    print("")

    print("         gamma  ", end="")
    for g in generators:
        print("%10f " % g.gamma, end="")
    print("")

    print("         zeta   ", end="")
    for g in generators:
        print("%10f " % g.zeta, end="")
    print("")

    print("         lambda ", end="")
    for g in generators:
        print("%10f " % g.l, end="")
    print("")

    print("--------------------------------------------------------------------------------")
    print("")

    print("power    Pmin   ", end="")
    for g in generators:
        print("%10f " % g.pmin, end="")
    print("")

    print("         Pmax   ", end="")
    for g in generators:
        print("%10f " % g.pmax, end="")
    print("")

    print("================================================================================")
    print("Losses:")
    for e in tuple[1]:
        for v in e:
            print("%12s " % v, end="")
        print("")

    print("================================================================================")
    print("Demand:    ", tuple[2])
    print("Max Total: ", tuple[3])



# v, a, b, c = generate_randomized_generators(7)
# toSave = []
# for i in v:
#     print()
#     for j in i.args():
#         print(str(j) + ",", end=" ")
