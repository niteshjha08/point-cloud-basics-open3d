#!/usr/bin/python3

## IMPORT LIBRARIES
import numpy as np
import time
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt

## Reference http://www.open3d.org/docs/release/tutorial/Basic/

# VISUALIZE THE POINT CLOUD
pcd = o3d.io.read_point_cloud('./test_files/KITTI/000000.pcd')
print(np.asanyarray(pcd.points))
o3d.visualization.draw_geometries([pcd])

# VOXEL GRID DOWNSAMPLING
print(f"Points before downsampling: {len(pcd.points)} ")
downpcd = pcd.voxel_down_sample(voxel_size = 0.1)
print(f"Points after downsampling: {len(downpcd.points)}")
o3d.visualization.draw_geometries([downpcd])

# RANSAC
plane_model, inliers = pcd.segment_plane(distance_threshold = 0.4, ransac_n = 3, num_iterations = 1000)
plane_cloud = pcd.select_by_index(inliers)
non_plane_cloud = pcd.select_by_index(inliers, invert = True)
o3d.visualization.draw_geometries([plane_cloud])

# DBScan CLUSTERING
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        pcd.cluster_dbscan(eps=0.2, min_points=10, print_progress=True))

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd])


# Bounding Boxes
bounding_boxes = []
inds =  pd.Series(range(len(labels))).groupby(labels, sort = False).apply(list).tolist()

for i in range(len(inds)):
    cluster = pcd.select_by_index(inds[i])
    bb = cluster.get_axis_aligned_bounding_box()
    bb.color = (1,0,0)
    bounding_boxes.append(bb)

visuals = []
visuals.append(pcd)
visuals.extend(bounding_boxes)
o3d.visualization.draw_geometries(visuals)