# adaptive_point_cloud_segmentation.py
#
# What this script does:
# It takes one point cloud and splits it into the ground and separate objects
# (buildings, trees, cars and so on). Most settings are worked out
# automatically from the data, which is why it is called "adaptive".
#
# Steps, in simple terms:
# 1. Load the point cloud (.ply, .pcd, .xyz or .txt).
# 2. Remove noise in two passes: points that sit far from their neighbours
#    (statistical filter) and points with too few neighbours nearby (radius
#    filter).
# 3. Find the ground with RANSAC (fitting a flat plane) and colour it green.
# 4. Group the non-ground points into objects with DBSCAN. The grouping
#    distance is based on the average spacing between points.
# 5. Drop DBSCAN noise, cluster again, and give each object its own colour.
# 6. Remove very small objects, cluster once more, and show the final result.
#
# A 3D viewer window opens after each main step (set visualize=False to skip).
# Inputs: one file path set at the bottom of the script (__main__ section).
# Needs: numpy, open3d, matplotlib, scikit-learn.

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import time

import random

random.seed(10)

def load_point_cloud(filename):
    print(f"Loading point cloud: {filename}")

    if filename.endswith('.pcd'):
        pcd = o3d.io.read_point_cloud(filename)
    elif filename.endswith('.ply'):
        pcd = o3d.io.read_point_cloud(filename)
    elif filename.endswith('.xyz') or filename.endswith('.txt'):
        points = np.loadtxt(filename, delimiter=' ')
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        if points.shape[1] >= 6:
            pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6] / 255.0)
    else:
        raise ValueError(f"Unsupported file format: {filename}")

    print(f"Point cloud loaded: {len(pcd.points)} points")
    return pcd

def estimate_point_cloud_density(pcd, sample_size=1000):
    points = np.asarray(pcd.points)

    if len(points) > sample_size:
        indices = np.random.choice(len(points), sample_size, replace=False)
        sample_points = points[indices]
    else:
        sample_points = points

    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    sum_dist = 0.0
    for point in sample_points:
        _, idx, dist = pcd_tree.search_knn_vector_3d(point, 2)
        sum_dist += np.sqrt(dist[1])

    avg_dist = sum_dist / len(sample_points)
    print(f"Estimated average point cloud density (distance): {avg_dist:.6f}")
    return avg_dist

def remove_statistical_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    print(f"Removing noise using statistical outlier removal...")
    start_time = time.time()

    # For older versions of Open3D, use the statistical_outlier_removal function
    cl, ind = o3d.geometry.PointCloud.remove_statistical_outlier(pcd,
                                                                nb_neighbors=nb_neighbors,
                                                                std_ratio=std_ratio)

    end_time = time.time()
    outlier_count = len(pcd.points) - len(cl.points)
    print(f"Statistical outlier removal time: {end_time - start_time:.2f} seconds")
    print(f"Removed {outlier_count} noise points ({outlier_count/len(pcd.points)*100:.2f}%)")

    return cl

def remove_radius_outliers(pcd, nb_points=16, radius=None):
    if radius is None:
        radius = estimate_point_cloud_density(pcd) * 3.0

    print(f"Removing noise using radius outlier removal...")
    print(f"Parameters: min_nb_points={nb_points}, radius={radius:.6f}")
    start_time = time.time()

    # For older versions of Open3D, use the radius_outlier_removal function
    cl, ind = o3d.geometry.PointCloud.remove_radius_outlier(pcd,
                                                          nb_points=nb_points,
                                                          radius=radius)

    end_time = time.time()
    outlier_count = len(pcd.points) - len(cl.points)
    print(f"Radius outlier removal time: {end_time - start_time:.2f} seconds")
    print(f"Removed {outlier_count} noise points ({outlier_count/len(pcd.points)*100:.2f}%)")

    return cl

def segment_ground_using_ransac(pcd, distance_threshold=0.1, ransac_n=3, num_iterations=100):
    print("Segmenting ground using RANSAC...")

    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    print(f"Point cloud dimensions: {extent}")

    max_dimension = np.max(extent)
    distance_threshold = max_dimension * 0.01
    print(f"Adaptive ground fitting distance threshold: {distance_threshold:.6f}")

    start_time = time.time()
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                         ransac_n=ransac_n,
                                         num_iterations=num_iterations)
    end_time = time.time()
    print(f"RANSAC ground fitting time: {end_time - start_time:.2f} seconds")

    [a, b, c, d] = plane_model
    print(f"Detected ground plane: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    print(f"Ground points count: {len(inliers)}")

    ground_points = pcd.select_by_index(inliers)
    non_ground_points = pcd.select_by_index(inliers, invert=True)

    ground_points.paint_uniform_color([0.0, 1.0, 0.0])

    return ground_points, non_ground_points

def adaptive_dbscan_clustering(pcd, eps_scale=5, min_points=200):
    print("Performing adaptive DBSCAN clustering...")

    avg_density = estimate_point_cloud_density(pcd)
    eps = avg_density * eps_scale
    points = np.asarray(pcd.points)

    min_points = max(min_points, int(np.log10(len(points))) * 15)
    print("len(points) is", len(points), "min_points is", min_points)

    print(f"Adaptive DBSCAN parameters: eps={eps:.6f}, min_points={min_points}")

    start_time = time.time()
    db = DBSCAN(eps=eps, min_samples=min_points, n_jobs=-1)
    labels = db.fit_predict(points)
    end_time = time.time()
    print(f"DBSCAN clustering time: {end_time - start_time:.2f} seconds")

    max_label = labels.max()
    noise_count = list(labels).count(-1)
    print(f"Detected clusters: {max_label + 1}")
    print(f"Noise points: {noise_count} ({noise_count/len(points)*100:.2f}%)")

    clusters_indices = []
    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]
        clusters_indices.append(cluster_indices)

    noise_indices = np.where(labels == -1)[0]
    if len(noise_indices) > 0:
        clusters_indices.append(noise_indices)

    return clusters_indices, labels

def remove_dbscan_noise(pcd, labels):
    print("Removing DBSCAN noise points...")

    non_noise_indices = np.where(labels != -1)[0]
    noise_indices = np.where(labels == -1)[0]

    cleaned_pcd = pcd.select_by_index(non_noise_indices)

    print(f"Removed {len(noise_indices)} noise points ({len(noise_indices)/len(pcd.points)*100:.2f}%)")

    return cleaned_pcd

def color_clusters(pcd, clusters_indices):
    colored_pcd = o3d.geometry.PointCloud()
    colored_pcd.points = pcd.points

    colors = np.zeros((len(pcd.points), 3))
    color_map = plt.get_cmap("tab20")

    for i, indices in enumerate(clusters_indices):
        if i < len(clusters_indices) - 1:  # Skip the last cluster if it's noise
            color_idx = i % 20
            cluster_color = color_map(color_idx)[:3]
            colors[indices] = cluster_color

    colored_pcd.colors = o3d.utility.Vector3dVector(colors)
    return colored_pcd

def filter_small_clusters(pcd, clusters_indices, labels, min_points_ratio=0.0005):
    total_points = len(pcd.points)
    min_points = int(total_points * min_points_ratio)
    print(f"Filtering small clusters: minimum points threshold={min_points}")

    valid_clusters = []
    valid_cluster_indices = []
    for i, indices in enumerate(clusters_indices):
        if len(indices) >= min_points:
            valid_clusters.append(i)
            valid_cluster_indices.extend(indices)

    filtered_pcd = pcd.select_by_index(valid_cluster_indices)

    new_labels = np.full(len(valid_cluster_indices), -1)
    start_idx = 0
    for new_i, old_i in enumerate(valid_clusters):
        cluster_size = len(clusters_indices[old_i])
        new_labels[start_idx:start_idx+cluster_size] = new_i
        start_idx += cluster_size

    print(f"Clusters before filtering: {len(clusters_indices)}")
    print(f"Clusters after filtering: {len(valid_clusters)}")
    print(f"Points before filtering: {total_points}")
    print(f"Points after filtering: {len(valid_cluster_indices)}")

    return filtered_pcd, new_labels

def adaptive_point_cloud_segmentation(filename, visualize=True):
    # 1. Load point cloud
    pcd = load_point_cloud(filename)

    # Visualize original point cloud
    if visualize:
        print("Displaying original point cloud...")
        o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")

    # 2. Remove noise using statistical outlier removal
    cleaned_pcd = remove_statistical_outliers(pcd, nb_neighbors=20, std_ratio=2.0)

    # 3. Remove noise using radius outlier removal
    cleaned_pcd = remove_radius_outliers(cleaned_pcd)

    # Visualize cleaned point cloud
    if visualize:
        print("Displaying cleaned point cloud...")
        o3d.visualization.draw_geometries([cleaned_pcd], window_name="Cleaned Point Cloud")

    # 4. Ground segmentation using RANSAC
    ground_pcd, non_ground_pcd = segment_ground_using_ransac(cleaned_pcd)

    # Visualize ground segmentation result
    if visualize:
        print("Displaying ground segmentation result...")
        o3d.visualization.draw_geometries([ground_pcd, non_ground_pcd],
                                      window_name="Ground Segmentation (green = ground)")

    # 5. Adaptive DBSCAN clustering on non-ground points
    clusters_indices, labels = adaptive_dbscan_clustering(non_ground_pcd)

    # 6. Remove DBSCAN noise points
    non_ground_cleaned = remove_dbscan_noise(non_ground_pcd, labels)

    # Recalculate clusters after noise removal
    clusters_indices, labels = adaptive_dbscan_clustering(non_ground_cleaned)

    # 7. Color each cluster
    colored_pcd = color_clusters(non_ground_cleaned, clusters_indices)

    # Visualize initial segmentation result
    if visualize:
        print("Displaying segmentation result...")
        o3d.visualization.draw_geometries([ground_pcd, colored_pcd],
                                      window_name="Segmentation Result")

    # 8. Filter small clusters
    filtered_pcd, filtered_labels = filter_small_clusters(non_ground_cleaned, clusters_indices, labels)

    # 9. Recalculate clusters and recolor
    refined_clusters_indices, refined_labels = adaptive_dbscan_clustering(filtered_pcd)
    refined_colored_pcd = color_clusters(filtered_pcd, refined_clusters_indices)

    # Visualize final result
    if visualize:
        print("Displaying final refined result...")
        o3d.visualization.draw_geometries([ground_pcd, refined_colored_pcd],
                                      window_name="Final Refined Result")

    return ground_pcd, refined_colored_pcd, refined_clusters_indices

if __name__ == "__main__":
    filename = "/Users/wanxy124/Downloads/ChangeDetection/section1/Aligned_2025_section1.ply"
    ground_pcd, segmented_pcd, clusters = adaptive_point_cloud_segmentation(filename)
    print(f"Complete! Detected {len(clusters)} objects.")
