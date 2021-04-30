"""
This notebook contains running examples of the 3 different depth-completion methods
  in this project.
"""
# %%
import numpy as np
import time

from matplotlib import pyplot as plt
import matplotlib.cm as cm
import PIL.Image as pil

from rvfn.utils.depth_completion import KnnDepth, MaskedConvDepth, \
    ModifiedIpBasicDepth



# %%
def visualize_depth_map(depth_map):
    height, width = depth_map.shape
    xs = []
    ys = []
    dists = []
    for i in range(height):
        for j in range(width):
            if depth_map[i][j] > 0:
                xs.append(j)
                ys.append(height - i)
                dists.append(depth_map[i][j])

    dists = np.clip(dists, 5, 70)-5

    fig, ax = plt.subplots(figsize=(16, 9), dpi=200,
                           linewidth=2, edgecolor='black')
    ax.axis('off')
    fig.gca().set_aspect('equal', adjustable='box')
    plt.scatter(xs, ys, c=dists, cmap='plasma', s=1.5)
    plt.close(fig)

    return fig


# %%
# depth_map_path = "data/depth_maps/001.npy" # single sweep
depth_map_path = "data/depth_maps/002.npy"  # 5 sweeps

depth_map = np.load(depth_map_path)

visualize_depth_map(depth_map)

# %%
# KNN depth completion

depth_knn = depth_map.copy()

start = time.time()

knnd = KnnDepth(depth_knn)
depth_knn = knnd.fill_depth()

end = time.time()
print(end - start)

visualize_depth_map(depth_knn)

# %%
# Masked convolution depth completion

depth_maskconv = depth_map.copy()

start = time.time()

mcd = MaskedConvDepth(depth_maskconv, depth_maskconv > 0)
depth_maskconv = mcd.fill_depth()

end = time.time()
print(end - start)

visualize_depth_map(depth_maskconv)

# %%
# IP-BASIC depth completion

depth_ipbasic = depth_map.copy()

start = time.time()

mipbd = ModifiedIpBasicDepth()
depth_ipbasic = mipbd.fill_depth(depth_ipbasic)

end = time.time()
print(end - start)

visualize_depth_map(depth_ipbasic)

# %%
