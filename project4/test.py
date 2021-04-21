import numpy as np
import em
import common

X = np.loadtxt("netflix_incomplete.txt")
X_gold = np.loadtxt("netflix_complete.txt")

K = 12

log_lh = [0, 0, 0, 0, 0]
best_seed = 0
mixtures = [0, 0, 0, 0, 0]
posts = [0, 0, 0, 0, 0]
rmse = 0.

# Test all seeds
for i in range(5):
    mixtures[i], posts[i], log_lh[i] = em.run(X, *common.init(X, K, i))

best_seed = np.argmax(log_lh)
Y = em.fill_matrix(X, mixtures[best_seed])
rmse = common.rmse(X_gold, Y)
print("RMSE for K = 12: {:.4f}".format(rmse))
