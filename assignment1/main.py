import sys
import numpy as np
import matplotlib.pyplot as plt


def generate_data (P, N):
    feature_vectors = np.transpose([np.random.standard_normal(P) for _ in range(N)])
    labels = np.random.choice([-1, 1], P)
    return labels, feature_vectors

def train(P, N, n_max):
    (labels, feature_vectors) = generate_data(P, N)
    epoch = 1

    weight_vector = np.zeros(N)

    while epoch <= n_max:
        changed_weight = False
        for i in range(P):
            datapoint = feature_vectors[i]
            label = labels[i]
            evt = np.dot(weight_vector, datapoint) * label
            if evt <= 0:
                changed_weight = True
                weight_vector = weight_vector + (1/N * datapoint * label)

        if not changed_weight:
            #print ("Finished training")
            return True

        epoch += 1

    #print ("Finished training because max epochs reached")
    return False


# P: Number of datapoints
# N: Number of features
# a: Value to infer P from using N
# n_D: Times we will run the training algorithm on a different dataset
# n_max: Maximum number of epochs
def repeat_training(a, N=20, n_D = 50, n_max = 100):
    P = int(a * N)
    print ("Repeating training for N={} P={} a={}".format(N, P, a))
    success_count = 0
    for i in range(n_D):
        if train(P, N, n_max):
            success_count += 1

    success_fraction = success_count / n_D
    print ("Computed success fraction:", success_fraction)
    return success_fraction


#print(generate_data(5, 3))
a_values = [a / 100 for a in range(75, 300, 25)]
success_fractions = [repeat_training(a) for a in a_values]

plt.plot(a_values, success_fractions)
plt.ylabel(r'$ Q_{l.s.}$')
plt.xlabel(r'$ \alpha = P/N$')
plt.title("Fraction of linearly separable dichotomies")
plt.show()
