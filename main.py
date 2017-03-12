#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This optimization problem deals with allocating loads to power generators
# of a plant for minimum total fuel cost and emissions while meeting
# the power demand and transmission losses constraints.

import math
import random
import yaml
from Firefly import *
from Generators import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib import cm

debug = False

#workaround for doubling first frame id
frames_counter = 0

# loads - power assigned to every power unit
# load - assigned power
# gen - generator

def single_gen_fuel_cost(load, gen):
    return gen.a + gen.b * load + gen.c * (load ** 2)


# f_1
def total_fuel_cost(loads, generators):
    f = 0
    for load, gen in zip(loads, generators):
        f += single_gen_fuel_cost(load, gen)

    return f


def single_gen_emissions(load, gen):
    emissions = 0

    emissions += math.pow(10, -2) * (gen.alpha + gen.beta * load
                                     + gen.gamma * (load ** 2))
    emissions += gen.zeta * math.exp(gen.l * load)

    return emissions


# f_2
def total_emissions(loads, generators):
    f = 0
    for load, gen in zip(loads, generators):
        f += single_gen_emissions(load, gen)

    return f


# f_3
def min_f_3(loads, loss_array, global_demand):
    toSub = 0
    n = len(loads)
    for i in range(0, n):
        for j in range(0, n):
            toSub += loss_array[i][j] * loads[i] * loads[j]

    minF = 0
    for i in range(0, n):
        minF += loads[i]

    return 1000 * abs(minF - global_demand - toSub)


def f_attractiveness(loads, generators, loss_array, global_demand, max_val):
    return (max_val - 0.5 * total_fuel_cost(loads, generators)
            - 0.5 * total_emissions(loads, generators)
            - min_f_3(loads, loss_array, global_demand))


def generateInitialPopulation(firefliesCount, generators, loss_array,
                              global_demand, max_val):
    fireflies = []
    for i in range(firefliesCount):
        firefly = Firefly(i, len(generators))

        for generator in generators:
            firefly.addRandomLoad(generator.pmin, generator.pmax)

        firefly.intensity = f_attractiveness(firefly.loads, generators,
                                             loss_array, global_demand, max_val)
        fireflies.append(firefly)

    return fireflies


def square_distance(loads_i, loads_j):
    sum = 0
    for i, j in zip(loads_i, loads_j):
        sum += math.pow(i - j, 2)

    return sum


def validate_ranges(ff, generators):
    loads = ff.loads
    changed = False
    for i in range(0, len(loads)):
        if (generators[i].pmin > loads[i]):
            loads[i] = generators[i].pmin
            changed = True

        if (generators[i].pmax < loads[i]):
            loads[i] = generators[i].pmax
            changed = True

    return changed


def print_results(generation, fireflies):
    bestVal = fireflies[0].intensity
    bestIndex = 0
    print("========== GENERATION {0} ==========".format(generation))
    for i in range(0, len(fireflies)):
        print("Firefile {0}: {1}".format(i, fireflies[i].intensity))
        if (fireflies[i].intensity > bestVal):
            bestVal = fireflies[i].intensity
            bestIndex = i

    print("\nBest firefly: {0}".format(bestVal))


def load_data(params_filename):
    with open(params_filename, 'r') as stream:
        return yaml.load(stream)


def print_algorithm_result(best, generators, loss_array, max_val):
    PLoss = 0
    for i in range(0, len(best.loads)):
        for j in range(0, len(best.loads)):
            PLoss += loss_array[i][j] * best.loads[i] * best.loads[j]

    print("")
    print("==================================== RESULTS ===================================")
    for i in range(0, len(best.loads)):
        print("PG%02d          (MW)     %16f" % (i + 1, best.loads[i]))

    print("Obj. function          %16f" % (max_val - best.intensity))
    print("Fuel cost     ($/hr)   %16f" % total_fuel_cost(best.loads, generators))
    print("Emission      (ton/hr) %16f" % total_emissions(best.loads, generators))
    print("PLoss                  %16f" % PLoss)


def ffalgorithm(fa_alpha, fa_beta, fa_gamma, fa_random_move_coefficient,
                MaxGeneration, firefliesCount, generators, loss_array,
                global_demand, max_val, preserve_best, delay, visualisation=False):
    best_ff_marker = None
    fireflies_markers = None
    fig = None
    ani = None
    if visualisation:
        fig = plt.figure()
        
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        # count axis scaling attributes
        min_loads = get_min_loads(generators)
        max_loads = get_max_loads(generators)

        min_cost = total_fuel_cost(min_loads, generators)
        max_cost = total_fuel_cost(max_loads, generators)

        min_emiss = .2 #total_emissions(min_loads, generators)
        max_emiss = .3 #total_emissions(max_loads, generators)

        print ('min cost: '  )
        print (min_cost      )
        print ('max cost: '  )
        print (max_cost      )
        print ('min emiss: ' )
        print (min_emiss     )
        print ('max emiss: ' )
        print (max_emiss     )

        width = max_cost - min_cost
        height = max_emiss - min_emiss

        # add plot
        ax = fig.add_subplot(111, autoscale_on=False, axisbg='#a2e2fe',
                             # TODO: This values shoudn't be hardcoded -> almost are
                             xlim=(min_cost, max_cost), ylim=(min_emiss, max_emiss))
        
        # init ids of fireflies points
        fireflies_ids = []
        for i in range(0, firefliesCount):
            fireflies_ids.append(ax.text(0, 0, '', fontsize=10))

        # add label with generation number info
        generation_label = ax.text(min_cost + 0.02 * (max_cost - min_cost), max_emiss - 0.05 * (max_emiss - min_emiss), 'Uruchamianie algorytmu...', fontsize=10)

        fireflies_markers = ax.scatter([], [], c=[], cmap=cm.Blues, s=50, label='frame', marker="o")
        best_ff_marker = ax.scatter([], [], c='r', s=50)

    # ----------------DATA--------------------
    # number of units power generators of a power plan
    n = len(generators)

    # ----------------DATA--------------------


    fireflies = generateInitialPopulation(firefliesCount, generators,
                                          loss_array, global_demand, max_val)
    if debug:
        for f in fireflies:
            print ('{0} total : ({1},{2})'.format(f.id, total_fuel_cost(f.loads, generators),
                total_emissions(f.loads, generators)))

    t = 0
    global_best_ff = fireflies[0]
    if visualisation:
        ani = animation.FuncAnimation(fig, ff_generation_step, frames=MaxGeneration,
                                      interval=delay, init_func=None, save_count=None,
                                      repeat=False,
                                      fargs=[firefliesCount, fireflies, fa_gamma,
                                             fa_beta, fa_alpha, generators,
                                             loss_array, global_demand, max_val,
                                             fa_random_move_coefficient, global_best_ff,
                                             n, t, preserve_best, fireflies_markers, best_ff_marker, 
                                             fireflies_ids, generation_label, width,
                                             height, visualisation])
        plt.show()
    else:
        while (t <= MaxGeneration):
            ff_generation_step(t, firefliesCount, fireflies, fa_gamma,
                               fa_beta, fa_alpha, generators,
                               loss_array, global_demand, max_val,
                               fa_random_move_coefficient, global_best_ff,
                               n, t, preserve_best)
            t += 1

    return global_best_ff, ani

def get_best(fireflies):
    fa_best = fireflies[0].intensity
    fa_best_ff = fireflies[0]
    for ff in fireflies:
        if (fa_best < ff.intensity):
            fa_best = ff.intensity
            fa_best_ff = ff
    return fa_best_ff

def get_worst_index(fireflies):
    fa_worst_ff = fireflies[0]
    fa_worst_ind = 0
    for i in range(0, len(fireflies)-1):
        if (fa_worst_ff.intensity > fireflies[i].intensity):
            fa_worst_ff = fireflies[i]
            fa_worst_ind = i
    return fa_worst_ind

def ff_generation_step(frame, firefliesCount, fireflies, fa_gamma,
                       fa_beta, fa_alpha, generators,
                       loss_array, global_demand, max_val,
                       fa_random_move_coefficient, global_best_ff,
                       n, t, preserve_best, fireflies_markers=None, 
                       best_ff_marker=None, fireflies_ids=None, generation_label=None, 
                       width=None, height=None, visualisation=False):
    for i in range(0, firefliesCount):
        for j in range(0, firefliesCount):
            if (fireflies[i].intensity >= fireflies[j].intensity):
                continue

            # moving i towards j
            fa_gravity = math.exp(-fa_gamma * square_distance(fireflies[i].loads,
                                                              fireflies[j].loads))

            for k in range(0, n):
                kth_load = fireflies[i].loads[k]
                fireflies[i].loads[k] = kth_load + (fa_beta * fa_gravity
                                                    * (fireflies[j].loads[k] - kth_load)
                                                    ) + fa_alpha * (random.random()
                                                                    - 0.5)

            # updating intensity
            fireflies[i].intensity = f_attractiveness(fireflies[i].loads,
                                                      generators,
                                                      loss_array,
                                                      global_demand,
                                                      max_val)

    # update out of range fireflies
    for ff in fireflies:
        # if loads changed -> upadate intensity
        if (validate_ranges(ff, generators)):
            if debug:
                print("Firefly {0} felt out of scope. Updating.".format(ff.id))
            ff.intensity = f_attractiveness(ff.loads, generators,
                                            loss_array, global_demand,
                                            max_val)
    # finding best firefly
    fa_best_ff = get_best(fireflies)
    fa_best = fa_best_ff.intensity

    #replace worst with copy of the best
    if preserve_best:
        worst_firefly_ind = get_worst_index(fireflies)
        worst_ff_id = fireflies[worst_firefly_ind].id
        fireflies[worst_firefly_ind] = fa_best_ff.deep_copy()
        fireflies[worst_firefly_ind].id = worst_ff_id
        #update so to move former worst instead of current best
        fa_best = fireflies[worst_firefly_ind]
        fa_best = fa_best_ff.intensity


    # perform random move of best firefly
    for k in range(0, n):
        kth_load = fa_best_ff.loads[k]
        fa_best_ff.loads[k] = kth_load + (fa_random_move_coefficient
                                          * (random.random() - 0.5)
                                          * (generators[k].pmax - generators[k].pmin))

    # handling firefly that went out of scope during random walk (what might have happened)
    if (validate_ranges(fa_best_ff, generators)):
            fa_best_ff.intensity = f_attractiveness(fa_best_ff.loads, generators,
                                            loss_array, global_demand,
                                            max_val)

    # updating best firefly
    fa_best_ff = get_best(fireflies)
    fa_best = fa_best_ff.intensity

    # save globally best firefly
    if (global_best_ff.intensity < fa_best_ff.intensity):
        global_best_ff = fa_best_ff.deep_copy()

    # printing result
    if (debug):
        print_results(t, fireflies)
    
    if visualisation:
        zipped = zip([total_fuel_cost(f.loads, generators) for f in fireflies],
                     [total_emissions(f.loads, generators) for f in fireflies])
        
        for i, (a, b) in enumerate(zipped):
            fireflies_ids[i-1].set_x(a + (0.005 * width))
            fireflies_ids[i-1].set_y(b + (0.005 * height))
            fireflies_ids[i-1].set_text('{0}'.format(fireflies[i-1].id))

        global frames_counter
        generation_label.set_text('Generacja: {0}'.format(frame))
        frames_counter+=1
        
        #should be optimized!
        zipped2 = zip([total_fuel_cost(f.loads, generators) for f in fireflies],
                     [total_emissions(f.loads, generators) for f in fireflies])

        fireflies_markers.set_offsets(list(sum(zipped2, ())))
        fireflies_markers.set_color(normalize_to_color([f.intensity for f in fireflies]))
        best_ff_marker.set_offsets(
            [total_fuel_cost(global_best_ff.loads, generators), total_emissions(global_best_ff.loads, generators)])
        return fireflies_markers


def normalize_to_color(intensity):
    m = min(intensity)
    c_list = list(intensity)
    if m < 0:
        c_list = [e + m for e in intensity]

    mi = min(c_list)
    ma = max(c_list)
    return [str((v - mi) / (ma - mi)) for v in c_list]

def get_max_loads(generators):
    max_loads = []
    for generator in generators:
        max_loads.append(generator.pmax)
    return max_loads

def get_min_loads(generators):
    min_loads = []
    for generator in generators:
        min_loads.append(generator.pmin)
    return min_loads

def main():
    data = load_data("resources/parameters.yml")

    alpha = data["alpha"]
    beta = data["beta"]
    gamma = data["gamma"]

    random_move_coefficient = data["random_move_coefficient"]

    max_generation = data["max_generation"]
    fireflies_count = data["fireflies_count"]

    visualisation = bool(data["visualisation"])

    preserve_best = bool(data["pass_best_to_next_generation"])

    delay = data["delay"]

    power_units = 6

    # generators
    # data_input = generators_from_file()
    data_input = generate_randomized_generators(6)
    generators = data_input[0]
    loss_array = data_input[1]  # B
    global_demand = data_input[2]  # D

    max_loads = get_max_loads(generators)

    max_val = total_fuel_cost(max_loads, generators) + total_emissions(max_loads, generators)

    best, ani = ffalgorithm(alpha, beta, gamma, random_move_coefficient,
                            max_generation, fireflies_count, generators, loss_array,
                            global_demand, max_val, preserve_best, delay, visualisation)

    print_algorithm_result(best, generators, loss_array, max_val)


main()
