# city_point_cloud_segmentation.py
#
# What this script does:
# It splits a city-scale point cloud into the ground and separate objects,
# and can save each part as its own file.
#
# Steps, in simple terms:
# 1. Load the point cloud and measure how dense it is (points per unit volume).
# 2. Use that density to choose the settings for the next steps.
# 3. Find the ground with RANSAC (fitting a flat plane) and colour it grey.
# 4. Throw away points that lie below the ground plane.
# 5. Group the remaining points into objects with DBSCAN clustering.
# 6. Keep only objects with at least 400 points; smaller ones count as noise.
# 7. Optionally save the ground, each object, and a combined coloured cloud.
#
# How to run it from a terminal:
#   python city_point_cloud_segmentation.py input.ply --output result.ply --visualize
# Needs: numpy, open3d, matplotlib, scikit-learn.

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import time
from sklearn.cluster import DBSCAN


def segment_point_cloud(pcd, eps=0.3, min_points=10):
    points = np.asarray(pcd.points)

    db = DBSCAN(eps=eps, min_samples=min_points).fit(points)
    labels = db.labels_

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Point cloud segmented into {num_clusters} clusters")

    clusters = []

    # Use a list of clearly distinct colours instead of a colormap
    distinct_colors = [
        [1, 0, 0],      # Red
        [0, 1, 0],      # Green
        [0, 0, 1],      # Blue
        [1, 1, 0],      # Yellow
        [1, 0, 1],      # Magenta
        [0, 1, 1],      # Cyan
        [0.5, 0, 0],    # Dark red
        [0, 0.5, 0],    # Dark green
        [0, 0, 0.5],    # Dark blue
        [0.5, 0.5, 0],  # Olive
        [0.5, 0, 0.5],  # Purple
        [0, 0.5, 0.5],  # Teal
        [1, 0.5, 0],    # Orange
        [0, 0.5, 1],    # Sky blue
        [1, 0, 0.5],    # Pink
    ]

    # Build a point cloud for each cluster and give it a colour
    for i in range(num_clusters):
        cluster_indices = np.where(labels == i)[0]
        cluster = pcd.select_by_index(cluster_indices)

        # Give each cluster a distinct colour
        color_idx = i % len(distinct_colors)
        cluster.paint_uniform_color(distinct_colors[color_idx])

        clusters.append(cluster)
        print(f"Cluster {i} contains {len(cluster.points)} points")

    # Colour the noise points grey
    noise_indices = np.where(labels == -1)[0]
    if len(noise_indices) > 0:
        noise_cluster = pcd.select_by_index(noise_indices)
        noise_cluster.paint_uniform_color([0.7, 0.7, 0.7])  # Grey
        clusters.append(noise_cluster)
        print(f"Noise contains {len(noise_indices)} points")

    return labels, num_clusters, clusters



def segment_ground_and_objects(input_cloud_path, distance_threshold=4,
                              ransac_n=3, num_iterations=1000,
                              dbscan_eps=1, min_points=300):
    print(f"Loading point cloud from {input_cloud_path}")
    pcd = o3d.io.read_point_cloud(input_cloud_path)

    if not pcd.has_points():
        raise ValueError("Point cloud contains no points!")

    print(f"Point cloud loaded with {len(pcd.points)} points")

    segments = {}
    segment_models = {}

    print("Detecting ground plane...")
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                           ransac_n=ransac_n,
                                           num_iterations=num_iterations)

    ground = pcd.select_by_index(inliers)
    non_ground = pcd.select_by_index(inliers, invert=True)

    a, b, c, d = plane_model
    print(f"Ground plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    print(f"Ground points: {len(ground.points)}, Non-ground points: {len(non_ground.points)}")

    ground.paint_uniform_color([0.5, 0.5, 0.5])
    segments['ground'] = ground
    segment_models['ground'] = plane_model

    # Remove points below ground plane
    print("Removing points below ground plane...")
    points = np.asarray(non_ground.points)
    colors = np.asarray(non_ground.colors)

    # Calculate distance to plane for each point
    # Plane equation: ax + by + cz + d = 0
    # Normalized distance: (ax + by + cz + d) / sqrt(a² + b² + c²)
    norm_factor = np.sqrt(a*a + b*b + c*c)
    distances = (points[:, 0] * a + points[:, 1] * b + points[:, 2] * c + d) / norm_factor

    # Keep only points above the ground plane (positive distance)
    above_ground_mask = distances > 0
    above_ground_indices = np.where(above_ground_mask)[0]

    non_ground = non_ground.select_by_index(above_ground_indices)
    print(f"After removing points below ground: {len(non_ground.points)} points")

    print("Clustering objects using DBSCAN...")
    # labels = np.array(non_ground.cluster_dbscan(eps=dbscan_eps, min_points=min_points))
    labels, num_clusters, clusters = segment_point_cloud(non_ground, dbscan_eps, min_points)
    o3d.visualization.draw_geometries([non_ground])


    if len(labels) == 0:
        print("No objects found after clustering!")
        return segments, segment_models, non_ground

    # Find valid cluster labels (exclude noise which is labeled as -1)
    unique_labels = np.unique(labels)
    valid_labels = unique_labels[unique_labels >= 0]

    print(f"Found {len(valid_labels)} potential clusters")

    # Check point count for each cluster and only keep clusters with > 400 points
    valid_objects = []
    for i in valid_labels:
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) >= 400:
            valid_objects.append(i)
            print(f"Object {i}: {len(cluster_indices)} points - VALID")
        else:
            print(f"Object {i}: {len(cluster_indices)} points - TOO SMALL (skipping)")

    print(f"Keeping {len(valid_objects)} clusters with at least 400 points")

    # Create colors for visualization
    max_label = labels.max() if labels.max() > 0 else 1
    colors = plt.get_cmap("tab20")(labels / max_label)
    colors[labels < 0] = [0, 0, 0, 1]  # Set noise points to black

    # Also set small clusters to black (treat as noise)
    for i in valid_labels:
        if i not in valid_objects:
            colors[labels == i] = [0, 0, 0, 1]

    non_ground.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # Only include valid objects in the segments dictionary
    for i in valid_objects:
        cluster_indices = np.where(labels == i)[0]
        segments[f'object_{i}'] = non_ground.select_by_index(cluster_indices)

    return segments, segment_models, non_ground

def adaptive_segment_city(input_cloud_path, output_path=None):
    start_time = time.time()

    pcd = o3d.io.read_point_cloud(input_cloud_path)
    points = np.asarray(pcd.points)

    # Calculate point cloud density to adaptively set parameters
    if len(points) > 0:
        # Compute point cloud bounds and size
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        dimensions = max_bound - min_bound
        volume = dimensions[0] * dimensions[1] * dimensions[2]

        # Calculate point density
        density = len(points) / volume if volume > 0 else 0

        # Determine a suitable distance threshold for RANSAC
        dist_threshold = 0.1 if density > 1000 else 0.2

        # Determine DBSCAN parameters based on density
        dbscan_eps = 0.5 / (density ** 0.25) if density > 0 else 0.5
        dbscan_eps = max(0.1, min(dbscan_eps, 2.0))  # Clamp between 0.1 and 2.0

        # Ensure minimum points is at least 400 as requested
        min_points = max(400, int(density * 0.01))
    else:
        # Default values if point cloud is empty
        dist_threshold = 0.2
        dbscan_eps = 0.5
        min_points = 400

    print(f"Adaptive parameters: RANSAC threshold={dist_threshold:.3f}, DBSCAN eps={dbscan_eps:.3f}, min_points={min_points}")

    segments, segment_models, labeled_cloud = segment_ground_and_objects(
        input_cloud_path,
        distance_threshold=dist_threshold,
        dbscan_eps=dbscan_eps,
        min_points=min_points
    )

    if output_path:
        for name, segment in segments.items():
            segment_path = output_path.replace('.ply', f'_{name}.ply')
            o3d.io.write_point_cloud(segment_path, segment)
            print(f"Saved segment '{name}' to {segment_path}")

        # Save colored point cloud
        full_segmented = segments['ground']
        for key, segment in segments.items():
            if key != 'ground':
                full_segmented += segment

        colored_output_path = output_path.replace('.ply', '_colored.ply')
        o3d.io.write_point_cloud(colored_output_path, full_segmented)
        print(f"Saved colored segmented point cloud to {colored_output_path}")

    elapsed_time = time.time() - start_time
    print(f"Segmentation completed in {elapsed_time:.2f} seconds")

    return segments, segment_models, labeled_cloud

def visualize_segments(segments, show_labels=True):
    geometries = []

    for name, segment in segments.items():
        if show_labels and len(segment.points) > 0:
            center = segment.get_center()
            label = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
            label.translate(center)
            geometries.append(label)

        geometries.append(segment)

    o3d.visualization.draw_geometries(geometries)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='City-scale point cloud segmentation')
    parser.add_argument('input', help='Input point cloud file path (.ply)')
    parser.add_argument('--output', help='Output directory for segmented point clouds', default=None)
    parser.add_argument('--visualize', action='store_true', help='Visualize the segmentation results')

    args = parser.parse_args()

    segments, segment_models, labeled_cloud = adaptive_segment_city(args.input, args.output)

    if args.visualize:
        to_visualize = [segments['ground'], labeled_cloud]
        o3d.visualization.draw_geometries(to_visualize)
