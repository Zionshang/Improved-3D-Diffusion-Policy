import numpy as np

def grid_sample_pcd(point_cloud, grid_size=0.005):
    """
    A simple grid sampling function for point clouds.

    Parameters:
    - point_cloud: A NumPy array of shape (N, 3) or (N, 6), where N is the number of points.
                   The first 3 columns represent the coordinates (x, y, z).
                   The next 3 columns (if present) can represent additional attributes like color or normals.
    - grid_size: Size of the grid for sampling.

    Returns:
    - A NumPy array of sampled points with the same shape as the input but with fewer rows.
    """
    coords = point_cloud[:, :3]  # Extract coordinates
    scaled_coords = coords / grid_size
    grid_coords = np.floor(scaled_coords).astype(int)
    
    # Create unique grid keys
    keys = grid_coords[:, 0] + grid_coords[:, 1] * 10000 + grid_coords[:, 2] * 100000000
    
    # Select unique points based on grid keys
    _, indices = np.unique(keys, return_index=True)
    
    # Return sampled points
    return point_cloud[indices]


def random_uniform_downsample(point_cloud, num_points):
    """
    Downsample (or pad) a point cloud to a fixed number of points using random
    uniform sampling.
    """
    num_coords = point_cloud.shape[1]
    if num_points > point_cloud.shape[0]:
        num_pad = num_points - point_cloud.shape[0]
        pad_points = np.zeros((num_pad, num_coords), dtype=point_cloud.dtype)
        sampled = np.concatenate([point_cloud, pad_points], axis=0)
    else:
        indices = np.random.choice(point_cloud.shape[0], num_points, replace=True)
        sampled = point_cloud[indices]

    return sampled


def color_weighted_downsample(point_cloud, num_points, target_color=(255, 255, 0), temperature=20.0):
    """
    根据颜色距离进行权重采样，越接近 target_color 的点被抽到的概率越高。
    
    - point_cloud: (N, 6)  [x,y,z,R,G,B]
    - num_points: 输出点数
    - target_color: (3,)  例如 [255, 255, 0]（黄色）
    - temperature: 颜色敏感度参数，越小越偏向目标颜色
    
    return:
        sampled_points_6d: (num_points, 6)
    """

    # 拆分
    colors = point_cloud[:, 3:6]  # (N,3)
    # 颜色距离（不归一化）
    color_dist = np.linalg.norm(colors - target_color, axis=1)  # (N,)
    # 相似度 → 颜色越近，值越大
    similarity = np.exp(-color_dist / temperature)
    # 归一化为采样概率
    probs = similarity / np.sum(similarity)
    # 按权重进行采样
    idx = np.random.choice(len(point_cloud), size=num_points, replace=len(point_cloud) < num_points, p=probs)

    return point_cloud[idx]


__all__ = ["grid_sample_pcd", "random_uniform_downsample", "color_weighted_downsample"]
