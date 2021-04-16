import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

######### Section 2: K-means ############
print("******* Section 2 *******\n ")
K = [1, 2, 3, 4] 
seeds = [0, 1, 2, 3, 4]

costs_kMeans = [0, 0, 0, 0, 0]

for k in range(len(K)):
    for i in range(len(seeds)):
        _, _, costs_kMeans[i] = kmeans.run(X, *common.init(X, K[k], seeds[i]))
        
    print("----- Clusters", k+1, " -----")
    print("Lowest cost: ", np.min(costs_kMeans))
    print("Best seed: ", np.argmin(costs_kMeans))

print("******* End of section 2 *******\n ")

######### Section 4: Comparing K-means and EM ############
print("******* Section 4 *******\n ")
costs_EM = [0, 0, 0, 0, 0]
mixtures_EM = [0, 0, 0, 0, 0] # Mixtures for best seed
bic = [0., 0., 0., 0.] # BIC for best cluster

for k in range(len(K)):
    for i in range(len(seeds)):
        mixtures_EM[i], _, costs_EM[i] = naive_em.run(X, *common.init(X, K[k], seeds[i]))
        
    bic[k] = common.bic(X, mixtures_EM[np.argmax(costs_EM)], np.max(costs_EM))

    print("----- Mixture ", k+1, " -----")
    print("Highest log: ", np.max(costs_EM))
    print("Best seed: ", np.argmax(costs_EM))

print("******* End of section 4 *******\n ")


######### Section 5: Bayesian Information Criterion ############
print("******* Section 5 *******\n ")
print("Best K: ", np.argmax(bic) + 1)
print("BIC for the best K: ", np.max(bic))

print("******* End of section 5 *******\n ")

######### Section 8: Using the mixture model for collaborative filtering ############
print("******* Section 8 *******\n ")
X = np.loadtxt("netflix_incomplete.txt")

K = [1, 12] 
seeds = [0, 1, 2, 3, 4]

costs_EM = [0, 0, 0, 0, 0]
mixtures_EM = [0, 0, 0, 0, 0] # Mixtures for best seed

for k in range(len(K)):
    for i in range(len(seeds)):
        mixtures_EM[i], _, costs_EM[i] = em.run(X, *common.init(X, K[k], seeds[i]))

    print("----- Mixture ", K[k], " -----")
    print("Highest log: ", np.max(costs_EM))
    print("Best seed: ", np.argmax(costs_EM))

print("******* End of section 8 *******\n ")
