import sys
import numpy as np
import matplotlib.pyplot as plt
import math


def magnitude(vec):
    return math.sqrt(sum([val**2 for val in vec]))

def generate_data (P, N):
    teacher_weights = np.random.rand(N)
    teacher_weights = teacher_weights * (math.sqrt(N) / magnitude(teacher_weights))
    feature_vectors = np.transpose([np.random.standard_normal(P) for _ in range(N)])
    labels = [np.dot(teacher_weights, x) for x in feature_vectors]
    return labels, feature_vectors, teacher_weights

# P: Number of datapoints
# N: Number of features
# a: Value to infer P from using N
# n_D: Times we will run the training algorithm on a different dataset
# n_max: Maximum number of epochs
def repeat_training(a, N=20, n_D = 50, n_max = 100):
    P = int(a * N)
    print ("Repeating training for N={} P={} a={}".format(N, P, a))
    success_count = 0
    total_gen_error = 0
    for i in range(n_D):
        total_gen_error += min_over(P, N, n_max)

    avg_gen_error = total_gen_error / n_D
    print ("Computed avg generalization error:", avg_gen_error)
    return avg_gen_error



def min_over(P, N, n_max):
    (labels, feature_vectors, teacher_vector) = generate_data(P, N)
    t = 0

    weight_vector = np.zeros(N)
    max_stability = 0

    while t <= n_max * P:
        min_stability = float("inf")
        for i in range(P):
            datapoint = feature_vectors[i]
            label = labels[i]
            stability = np.dot(weight_vector, datapoint) * label
            if stability <= min_stability:
                min_stability = stability
                min_index = i
        
        min_datapoint = feature_vectors[min_index]
        min_label = labels[min_index]
        weight_vector = weight_vector + (1/N * min_datapoint * min_label)
        
        t += 1
    
    upper = np.dot(weight_vector, teacher_vector)
    lower = magnitude(weight_vector) * magnitude(teacher_vector)
    gen_error = 1 / math.pi * math.acos(upper / lower)

    #print ("Finished training because max epochs reached")
    return gen_error
    
    

#print(generate_data(5, 3))
a_values = [a / 100 for a in range(25, 300, 25)]
gen_errors = [repeat_training(a) for a in a_values]

plt.plot(a_values, gen_errors)
plt.ylabel(r'$ \epsilon _g(t_{max})$')
plt.xlabel(r'$ \alpha = P/N$')
plt.title("Generalization error")
plt.show()
