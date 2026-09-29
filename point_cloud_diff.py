# point_cloud_diff.py
#
# What this script does:
# It compares two point clouds of the same place captured at different times
# (for example 2025 and 2021) and finds the parts that have changed.
#
# Steps, in simple terms:
# 1. Load the two point clouds.
# 2. Shift each cloud up or down so its lowest point sits at height zero.
# 3. Thin out the points (voxel downsampling) so the rest runs faster.
# 4. Find the ground with RANSAC (fitting a flat plane) and remove it.
# 5. Group the remaining points into objects with DBSCAN clustering.
# 6. Describe each object (size, centre, point count) and pair up objects
#    that look alike in the two clouds.
# 7. Inside each matched pair, look only at X and Y (ignoring height) and keep
#    points that have no partner within a set distance. These are changes.
# 8. Clean up the change points by dropping small, scattered bits, then keep
#    the largest connected change region.
# 9. Show the result: red = present only in cloud 1, blue = present only in
#    cloud 2, grey = the rest of cloud 2.
#
# Inputs: two file paths set at the bottom of the script (__main__ section).
# Needs: numpy, open3d, matplotlib, scikit-learn.

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import copy
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree

def load_point_clouds(file1, file2):
    pcd1 = o3d.io.read_point_cloud(file1)
    pcd2 = o3d.io.read_point_cloud(file2)

    print(f"Point cloud 1 contains {len(pcd1.points)} points")
    print(f"Point cloud 2 contains {len(pcd2.points)} points")

    return pcd1, pcd2

def remove_ground_ransac(pcd, distance_threshold=0.1, ransac_n=3, num_iterations=1000):
    print("Using RANSAC to detect ground plane...")

    points = np.asarray(pcd.points)

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )

    [a, b, c, d] = plane_model
    print(f"Detected plane: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    normal_vector = np.array([a, b, c])
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    cos_angle = np.abs(np.dot(normal_vector, np.array([0, 0, 1])))

    if cos_angle < 0.7:
        print("Warning: Detected plane may not be ground, angle with z-axis is large")

    ground_pcd = pcd.select_by_index(inliers)
    non_ground_pcd = pcd.select_by_index(inliers, invert=True)

    ground_pcd.paint_uniform_color([0.8, 0.8, 0.8])

    print(f"Ground points: {len(ground_pcd.points)}")
    print(f"Non-ground points: {len(non_ground_pcd.points)}")

    return non_ground_pcd, ground_pcd

def visualize_ground_removal(original_pcd, non_ground_pcd, ground_pcd):
    print("Original point cloud:")
    o3d.visualization.draw_geometries([original_pcd])

    # print("Ground points:")
    # o3d.visualization.draw_geometries([ground_pcd])
    #
    # print("Non-ground points:")
    # o3d.visualization.draw_geometries([non_ground_pcd])

    vis_pcd = copy.deepcopy(original_pcd)
    vis_ground = copy.deepcopy(ground_pcd)
    vis_non_ground = copy.deepcopy(non_ground_pcd)

    print("Ground and non-ground points:")
    o3d.visualization.draw_geometries([vis_ground, vis_non_ground])

def downsample_point_clouds(pcd1, pcd2, voxel_size=0.05):
    print(f"Downsampling with voxel size {voxel_size}")

    pcd1_down = pcd1.voxel_down_sample(voxel_size)
    pcd2_down = pcd2.voxel_down_sample(voxel_size)

    print(f"After downsampling, point cloud 1 contains {len(pcd1_down.points)} points")
    print(f"After downsampling, point cloud 2 contains {len(pcd2_down.points)} points")

    return pcd1_down, pcd2_down

def segment_point_cloud(pcd, eps=0.3, min_points=10):
    points = np.asarray(pcd.points)

    db = DBSCAN(eps=eps, min_samples=min_points).fit(points)
    labels = db.labels_

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Point cloud segmented into {num_clusters} clusters")

    clusters = []

    distinct_colors = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [0.5, 0, 0],
        [0, 0.5, 0],
        [0, 0, 0.5],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5],
        [1, 0.5, 0],
        [0, 0.5, 1],
        [1, 0, 0.5],
    ]

    for i in range(num_clusters):
        cluster_indices = np.where(labels == i)[0]
        cluster = pcd.select_by_index(cluster_indices)

        color_idx = i % len(distinct_colors)
        cluster.paint_uniform_color(distinct_colors[color_idx])

        clusters.append(cluster)
        print(f"Cluster {i} contains {len(cluster.points)} points")

    noise_indices = np.where(labels == -1)[0]
    if len(noise_indices) > 0:
        noise_cluster = pcd.select_by_index(noise_indices)
        noise_cluster.paint_uniform_color([0.7, 0.7, 0.7])
        clusters.append(noise_cluster)
        print(f"Noise contains {len(noise_indices)} points")

    return labels, num_clusters, clusters

def compute_cluster_features(clusters):
    features = []

    for cluster in clusters:
        num_points = len(cluster.points)
        centroid = np.mean(np.asarray(cluster.points), axis=0)
        bbox = cluster.get_axis_aligned_bounding_box()
        bbox_size = bbox.get_max_bound() - bbox.get_min_bound()
        volume = np.prod(bbox_size)

        feature = {
            'num_points': num_points,
            'centroid': centroid,
            'bbox_size': bbox_size,
            'volume': volume
        }

        features.append(feature)

    return features

def match_clusters(features1, features2, threshold=0.2):
    matches = []

    for i, feat1 in enumerate(features1):
        best_match = None
        best_score = float('inf')

        for j, feat2 in enumerate(features2):
            centroid_dist = np.linalg.norm(feat1['centroid'] - feat2['centroid'])
            volume_diff = abs(feat1['volume'] - feat2['volume']) / max(feat1['volume'], feat2['volume'])
            points_diff = abs(feat1['num_points'] - feat2['num_points']) / max(feat1['num_points'], feat2['num_points'])
            score = centroid_dist + volume_diff + points_diff

            if score < best_score:
                best_score = score
                best_match = j

        if best_score < threshold:
            matches.append((i, best_match, best_score))

    return matches

def detect_differences(pcd1, pcd2, clusters1, clusters2, matches, distance_threshold=0.01, xy_threshold=1.0):
    diff_clusters1 = []
    diff_clusters2 = []

    for i, j, _ in matches:
        cluster1 = clusters1[i]
        cluster2 = clusters2[j]

        points1 = np.asarray(cluster1.points)
        points2 = np.asarray(cluster2.points)

        diff_points1 = []
        diff_points2 = []

        points2_xy = points2[:, 0:2]
        tree2_xy = KDTree(points2_xy)

        for idx1, point1 in enumerate(points1):
            point1_xy = point1[0:2]

            dists, indices = tree2_xy.query([point1_xy], k=1)

            if dists[0][0] > xy_threshold:
                diff_points1.append(idx1)

        points1_xy = points1[:, 0:2]
        tree1_xy = KDTree(points1_xy)

        for idx2, point2 in enumerate(points2):
            point2_xy = point2[0:2]

            dists, indices = tree1_xy.query([point2_xy], k=1)

            if dists[0][0] > xy_threshold:
                diff_points2.append(idx2)

        if diff_points1:
            diff_cluster1 = cluster1.select_by_index(diff_points1)
            diff_cluster1.paint_uniform_color([1, 0, 0])
            diff_clusters1.append(diff_cluster1)
            print(f"Found {len(diff_points1)} difference points in cluster1-{i} (added)")

        if diff_points2:
            diff_cluster2 = cluster2.select_by_index(diff_points2)
            diff_cluster2.paint_uniform_color([0, 0, 1])
            diff_clusters2.append(diff_cluster2)
            print(f"Found {len(diff_points2)} difference points in cluster2-{j} (removed)")

    return diff_clusters1, diff_clusters2

def filter_isolated_points(diff_clusters, min_neighbors=5, radius=2.0):
    filtered_clusters = []

    for cluster in diff_clusters:
        points = np.asarray(cluster.points)
        if len(points) < 10:
            continue

        tree = KDTree(points)

        neighbors_count = tree.query_radius(points, r=radius, count_only=True)

        dense_indices = np.where(neighbors_count >= min_neighbors)[0]

        if len(dense_indices) > 0:
            dense_cluster = cluster.select_by_index(dense_indices)

            if len(dense_cluster.points) > 20:
                dense_cluster.paint_uniform_color(np.asarray(cluster.colors)[0])
                filtered_clusters.append(dense_cluster)

    return filtered_clusters

def visualize_clusters(pcd, clusters, window_name="Cluster Visualization"):
    geometries = [pcd] + clusters
    o3d.visualization.draw_geometries(geometries, window_name=window_name)

def visualize_differences(pcd1, pcd2, diff_clusters1, diff_clusters2):
    common_points = copy.deepcopy(pcd2)
    # common_points.paint_uniform_color([0.8, 0.8, 0.8])

    vis_objects = [common_points]

    for cluster in diff_clusters1:
        cluster.paint_uniform_color([1, 0, 0])
        vis_objects.append(cluster)

    for cluster in diff_clusters2:
        cluster.paint_uniform_color([0, 0, 1])
        vis_objects.append(cluster)

    print("Visualizing point cloud differences (considering only XY plane):")
    print(" - Red: Parts added in point cloud 1 (not in point cloud 2)")
    print(" - Blue: Parts added in point cloud 2 (not in point cloud 1)")
    print(" - Gray: Common parts in both point clouds (regardless of Z difference)")

    o3d.visualization.draw_geometries(
        vis_objects,
        window_name="Point Cloud Differences: Red-Added, Blue-Removed, Gray-Common"
    )

def align_point_clouds_to_ground(pcd1, pcd2):
    aligned_pcd1 = copy.deepcopy(pcd1)
    aligned_pcd2 = copy.deepcopy(pcd2)

    points1 = np.asarray(aligned_pcd1.points)
    points2 = np.asarray(aligned_pcd2.points)

    min_z1 = np.min(points1[:, 2])
    min_z2 = np.min(points2[:, 2])

    print(f"Minimum Z coordinate in point cloud 1: {min_z1}")
    print(f"Minimum Z coordinate in point cloud 2: {min_z2}")

    translation1 = np.array([0, 0, -min_z1])
    translation2 = np.array([0, 0, -min_z2])

    points1 = points1 + translation1
    points2 = points2 + translation2

    aligned_pcd1.points = o3d.utility.Vector3dVector(points1)
    aligned_pcd2.points = o3d.utility.Vector3dVector(points2)

    print(f"Point cloud 1 translated {translation1[2]} units to ground")
    print(f"Point cloud 2 translated {translation2[2]} units to ground")

    return aligned_pcd1, aligned_pcd2

def merge_clusters_by_distance(clusters, distance_threshold):
    """
    Merge clusters that are close to each other using a region growing approach.

    Args:
        clusters: List of point cloud clusters
        distance_threshold: Maximum distance for clusters to be merged

    Returns:
        List of merged clusters
    """
    if not clusters:
        return []

    # Convert all clusters to a unified point cloud format for processing
    merged_clusters = []

    # Extract all points and their original cluster indices
    all_points = []
    point_to_cluster = []

    for i, cluster in enumerate(clusters):
        points = np.asarray(cluster.points)
        all_points.append(points)
        point_to_cluster.extend([i] * len(points))

    # Combine all points into a single array
    if all_points:
        all_points = np.vstack(all_points)
        point_to_cluster = np.array(point_to_cluster)
    else:
        return []

    # Build KD-tree for efficient nearest neighbor search
    tree = KDTree(all_points)

    # Initialize cluster labels for connected component analysis
    cluster_labels = np.full(len(all_points), -1)
    current_label = 0

    # Process each point
    for i in range(len(all_points)):
        if cluster_labels[i] != -1:
            continue

        # Start a new cluster
        cluster_labels[i] = current_label

        # Use a queue for region growing
        queue = [i]
        while queue:
            point_idx = queue.pop(0)
            point = all_points[point_idx]

            # Find neighbors within distance threshold
            indices = tree.query_radius([point], r=distance_threshold)[0]

            for neighbor_idx in indices:
                if cluster_labels[neighbor_idx] == -1:
                    cluster_labels[neighbor_idx] = current_label
                    queue.append(neighbor_idx)

        current_label += 1

    # Create new clusters based on merged labels
    print(f"Merged into {current_label} connected components")

    # Count points in each merged cluster
    cluster_sizes = []
    for label in range(current_label):
        cluster_sizes.append(np.sum(cluster_labels == label))

    # Find the largest cluster
    if cluster_sizes:
        largest_cluster_label = np.argmax(cluster_sizes)
        largest_cluster_indices = np.where(cluster_labels == largest_cluster_label)[0]

        # Create point cloud for the largest cluster
        largest_cluster_points = all_points[largest_cluster_indices]
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)

        # Color it red
        largest_cluster_pcd.paint_uniform_color([1, 0, 0])

        print(f"Largest connected component has {len(largest_cluster_indices)} points")
        merged_clusters.append(largest_cluster_pcd)

    return merged_clusters

def extract_largest_region(diff_clusters, region_growing_threshold):
    """
    Apply region growing to merge clusters and extract the largest connected component

    Args:
        diff_clusters: List of difference clusters
        region_growing_threshold: Distance threshold for region growing

    Returns:
        List containing only the largest connected component
    """
    print(f"Starting region growing with threshold {region_growing_threshold}...")

    # Apply region growing to merge clusters
    merged_clusters = merge_clusters_by_distance(diff_clusters, region_growing_threshold)

    print(f"After region growing, found {len(merged_clusters)} connected components")

    return merged_clusters

def combine_clusters(clusters):
    """
    Combine multiple clusters into a single point cloud

    Args:
        clusters: List of point cloud clusters

    Returns:
        Combined point cloud
    """
    if not clusters:
        return None

    # Combine all points
    all_points = []
    all_colors = []

    for cluster in clusters:
        points = np.asarray(cluster.points)
        colors = np.asarray(cluster.colors)

        all_points.append(points)
        all_colors.append(colors)

    # Combine into single arrays
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)

    # Create new point cloud
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(all_points)
    combined_pcd.colors = o3d.utility.Vector3dVector(all_colors)

    return combined_pcd

def find_points_in_base_cloud(diff_clusters, base_cloud, distance_threshold=0.01):
    """
    Find the indices of points in the base cloud that correspond to the points in the difference clusters

    Args:
        diff_clusters: List of difference clusters (containing the red points)
        base_cloud: The original point cloud (pcd1_no_ground)
        distance_threshold: Distance threshold for point matching

    Returns:
        List of indices in the base cloud that match the diff clusters
    """
    # Extract points from base cloud
    base_points = np.asarray(base_cloud.points)

    # Build KD-tree for the base cloud
    base_tree = KDTree(base_points)

    # Collect all points from the difference clusters
    diff_points = []
    for cluster in diff_clusters:
        diff_points.append(np.asarray(cluster.points))

    if not diff_points:
        return []

    diff_points = np.vstack(diff_points)

    # Find the closest points in the base cloud
    distances, indices = base_tree.query(diff_points, k=1)

    # Keep only the points that are within the threshold
    valid_indices = indices[distances[:, 0] <= distance_threshold]

    # Remove duplicates
    unique_indices = np.unique(valid_indices)

    print(f"Found {len(unique_indices)} corresponding points in the base cloud")
    return unique_indices

def grow_regions_in_base_cloud(base_cloud, seed_indices, region_growing_threshold, min_points):
    if len(seed_indices) == 0:
        return []

    # Extract all points from the base cloud
    all_points = np.asarray(base_cloud.points)

    # Build KD-tree for the base cloud
    tree = KDTree(all_points)

    # Initialize labels for connected component analysis
    # -2: not a seed point, -1: seed point but not processed, 0+: assigned to a region
    labels = np.full(len(all_points), -2)

    # Mark seed points
    labels[seed_indices] = -1

    current_label = 0

    # Process each seed point
    for i in seed_indices:
        if labels[i] != -1:  # Already processed
            continue

        # Start a new region
        labels[i] = current_label

        # Use a queue for region growing
        queue = [i]
        while queue:
            point_idx = queue.pop(0)
            point = all_points[point_idx]

            # Find neighbors within distance threshold
            indices = tree.query_radius([point], r=region_growing_threshold)[0]

            for neighbor_idx in indices:
                # Only grow into points that are not yet assigned to a region
                if labels[neighbor_idx] < 0:  # Either -2 or -1
                    labels[neighbor_idx] = current_label
                    queue.append(neighbor_idx)

        current_label += 1

    print(f"Grown {current_label} regions in the base cloud")

    # Find the largest region
    if current_label > 0:
        region_sizes = []
        for label in range(current_label):
            region_sizes.append(np.sum(labels == label))

        # Get the label of the largest region
        largest_region_label = np.argmax(region_sizes)
        largest_region_size = region_sizes[largest_region_label]

        print(f"Largest region has {largest_region_size} points")

        # Filter out regions that are too small
        if largest_region_size >= min_points:
            # Extract points for the largest region
            largest_region_indices = np.where(labels == largest_region_label)[0]
            largest_region_points = all_points[largest_region_indices]

            # Create a new point cloud for the largest region
            largest_region_cloud = o3d.geometry.PointCloud()
            largest_region_cloud.points = o3d.utility.Vector3dVector(largest_region_points)
            largest_region_cloud.paint_uniform_color([1, 0, 0])  # Red color

            return [largest_region_cloud]

    return []

def identify_and_grow_main_structure(diff_clusters, base_cloud, distance_threshold=0.01,
                                     region_growing_threshold=5.0, min_points=100):

    print("Finding corresponding points in the base cloud...")
    seed_indices = find_points_in_base_cloud(diff_clusters, base_cloud, distance_threshold)

    print("Growing regions from seed points...")
    grown_regions = grow_regions_in_base_cloud(base_cloud, seed_indices,
                                               region_growing_threshold, min_points)

    return grown_regions

def visualize_main_structure(base_cloud, structure_cloud):
    """
    Visualize the main structure within the base cloud

    Args:
        base_cloud: The original point cloud
        structure_cloud: Point cloud of the main structure
    """
    # Create a copy of the base cloud with gray color
    vis_base = copy.deepcopy(base_cloud)
    vis_base.paint_uniform_color([0.8, 0.8, 0.8])

    # Prepare visualization objects
    vis_objects = [vis_base]

    # Add the structure cloud
    for cloud in structure_cloud:
        vis_objects.append(cloud)

    print("Visualizing the main structure (red) within the original point cloud (gray):")
    o3d.visualization.draw_geometries(
        vis_objects,
        window_name="Main Structure in Original Point Cloud"
    )

def main(file1, file2, voxel_size=0.05, eps=0.3, min_points=10, match_threshold=0.2, diff_threshold=0.01,
         xy_threshold=1.0, ground_distance=0.1, ransac_n=3, ransac_iterations=1000):
    pcd1, pcd2 = load_point_clouds(file1, file2)

    pcd1, pcd2 = align_point_clouds_to_ground(pcd1, pcd2)

    pcd1_down, pcd2_down = downsample_point_clouds(pcd1, pcd2, voxel_size)

    pcd1_no_ground, ground_pcd1 = remove_ground_ransac(
        pcd1_down,
        distance_threshold=ground_distance,
        ransac_n=ransac_n,
        num_iterations=ransac_iterations
    )

    pcd2_no_ground, ground_pcd2 = remove_ground_ransac(
        pcd2_down,
        distance_threshold=ground_distance,
        ransac_n=ransac_n,
        num_iterations=ransac_iterations
    )

    print("Visualizing ground removal results for point cloud 1:")
    visualize_ground_removal(pcd1_down, pcd1_no_ground, ground_pcd1)

    print("Visualizing ground removal results for point cloud 2:")
    visualize_ground_removal(pcd2_down, pcd2_no_ground, ground_pcd2)

    labels1, num_clusters1, clusters1 = segment_point_cloud(pcd1_no_ground, eps, min_points)
    labels2, num_clusters2, clusters2 = segment_point_cloud(pcd2_no_ground, eps, min_points)

    features1 = compute_cluster_features(clusters1)
    features2 = compute_cluster_features(clusters2)

    matches = match_clusters(features1, features2, match_threshold)
    print(f"Found {len(matches)} matched cluster pairs")

    diff_clusters1, diff_clusters2 = detect_differences(
        pcd1_no_ground, pcd2_no_ground, clusters1, clusters2, matches,
        distance_threshold=diff_threshold, xy_threshold=xy_threshold
    )

    print("Applying filters to remove isolated points...")
    diff_clusters1 = filter_isolated_points(diff_clusters1, min_neighbors=50, radius=3.0)
    diff_clusters1_filtered = filter_isolated_points(diff_clusters1, min_neighbors=50, radius=5.0)
    diff_clusters2 = filter_isolated_points(diff_clusters2, min_neighbors=50, radius=3.0)
    diff_clusters2 = filter_isolated_points(diff_clusters2, min_neighbors=50, radius=5.0)

    diff_clusters1_final = extract_largest_region(diff_clusters1_filtered, region_growing_threshold=10.0)

    main_structure = identify_and_grow_main_structure(
        diff_clusters1_final,
        pcd1_no_ground,
        distance_threshold=1,  # Threshold for matching points
        region_growing_threshold=1.0,  # Distance for region growing (adjust as needed)
        min_points=200  # Minimum points to keep a region
    )

    # Visualize the results
    # print("Visualizing the main structure within the original point cloud:")
    # visualize_main_structure(pcd1_no_ground, main_structure)


    print("Visualizing differences with isolated points removed:")
    visualize_differences(pcd1_no_ground, pcd2, diff_clusters1_final, diff_clusters2)

    return diff_clusters1_final, diff_clusters2

if __name__ == "__main__":
    point_cloud1 = "/Users/wanxy124/Downloads/ChangeDetection/section5/Aligned_2025_section5.ply"
    point_cloud2 = "/Users/wanxy124/Downloads/ChangeDetection/section5/Aligned_2021_section5.ply"

    diff1, diff2 = main(
        point_cloud1,
        point_cloud2,
        voxel_size=1,
        eps=3,
        min_points=200,
        match_threshold=30,
        diff_threshold=0.1,
        xy_threshold=1.0,
        ground_distance=0.5,
        ransac_n=3,
        ransac_iterations=1000
    )
