import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

out_dir = sys.argv[1]
results_file = sys.argv[2]
save_as = sys.argv[3]

path = os.path.join(out_dir,results_file)
save_path = os.path.join(out_dir,save_as)

with open(path, 'rb') as f:
    results = pickle.load(f)
results = np.array(results)

fig, ax = plt.subplots(figsize=(8,8))
ax.plot(results[:,0],results[:,1],label='train loss')
# ax.plot(results[:,0],results[:,2],label='val loss')
ax.set_xlabel('Iterations')
ax.set_ylabel('Loss')
ax.legend()
ax.grid()

plt.savefig(save_path)