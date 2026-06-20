import numpy as np
import open3d as o3d
from skimage.draw import polygon

data = np.load("data/sample_0/scene.npz")
points: np.ndarray = data["points"]
box_p1, box_p2, box_p3, box_p4 = (
    points[595, 1400],
    points[663, 1685],
    points[709, 1677],
    points[602, 1718],
)
real_p1_p2 = 0.78
scale = real_p1_p2 / np.linalg.norm(box_p1 - box_p2)
scaled_points = points * scale

mat_p1, mat_p2, mat_p3, mat_p4 = ((440, 1420), (530, 1940), (1536, 1800), (1200, 450))
corners = np.array([mat_p1, mat_p2, mat_p3, mat_p4])
rr, cc = polygon(corners[:, 0], corners[:, 1], shape=points.shape[:2])
mask = np.zeros(points.shape[:2], dtype=bool)
mask[rr, cc] = True

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(scaled_points[mask])

pcd = pcd.voxel_down_sample(voxel_size=0.01)
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd.estimate_normals()

# radii = [0.014, 0.028, 0.056, 0.112, 0.224]
# rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#     pcd, o3d.utility.DoubleVector(radii)
# )
# o3d.visualization.draw_geometries([pcd, rec_mesh])  # type: ignore

# mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=7)
# mesh.paint_uniform_color([0.7, 0.7, 0.7])
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh])  # type: ignore

pcd.orient_normals_towards_camera_location([0, 0, 0])
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=11, n_threads=1  # set n_threads to 1 (deterministic solve)
)
# o3d.io.write_triangle_mesh("data/sample_0/scene.obj", mesh)

mesh.paint_uniform_color([0.7, 0.7, 0.7])
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])  # type: ignore
