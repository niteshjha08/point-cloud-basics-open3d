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


