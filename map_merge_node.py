import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid

import numpy as np
import cv2

from map_merge import match_with_multi_level_features, adaptive_warp_and_merge

class MapMerge(Node):
    def __init__(self):
        super().__init__("MapMerge")
        self.map1 = None
        self.map2 = None
        self.merged_map = None

        self.map1_sub_ = self.create_subscription(OccupancyGrid, "/robot_0/map", lambda msg: self.map_callback(msg, map_id=1), 5)
        self.map2_sub_ = self.create_subscription(OccupancyGrid, "/robot_1/map", lambda msg: self.map_callback(msg, map_id=2), 5)

        self.merged_map_pub_ = self.create_publisher(OccupancyGrid, "/merged_map", 5)

    def map_callback(self, msg: OccupancyGrid, map_id: int):
        width = msg.info.width
        height = msg.info.height
        occupancy_array = np.array(msg.data, dtype=np.int8).reshape((height, width))
        flipped_map = np.flipud(occupancy_array)
        flipped_map[flipped_map <= 50] = 0
        flipped_map[flipped_map > 50] = 255

        flipped_map = cv2.bitwise_not(flipped_map)
        
        if map_id == 1:
            self.map1 = flipped_map
        elif map_id == 2:
            self.map2 = flipped_map

        self.merge()

    def merge(self):
        
        if not self.map1 or not self.map2:
            return
        
        H, kp1, kp2, matches, feature_type, filtered_matches, inliers = match_with_multi_level_features(self.map1, self.map2)

        if H is None:
            print(f"  - 无法找到有效的变换矩阵，请检查图像是否有足够的共同特征")

        # 可视化匹配结果
        if filtered_matches is not None and len(filtered_matches) > 0:
            
            try:
                # 从filtered_matches和inliers创建可视化掩码
                vis_mask = np.zeros(len(filtered_matches), dtype=np.uint8)
                
                # 获取匹配点坐标用于验证
                src_pts = np.float32([kp1[m.queryIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)
                
                # 变换源点并计算与目标点的距离
                transformed_pts = cv2.transform(src_pts, H[:2])
                
                # 计算内点比例作为融合权重参考
                for i, (dst_pt, transformed_pt) in enumerate(zip(dst_pts.reshape(-1, 2), transformed_pts.reshape(-1, 2))):
                    # 计算欧氏距离
                    distance = np.linalg.norm(transformed_pt - dst_pt)
                    # 使用与RANSAC相同的阈值
                    if distance < 3.0:
                        vis_mask[i] = 1
                
                # 计算内点比例
                inlier_ratio = np.sum(vis_mask) / len(vis_mask) if len(vis_mask) > 0 else 0.5
                blend_weight = max(0.3, min(0.7, inlier_ratio))  # 限制在0.3-0.7之间
                print(f"  - 可视化验证的内点比例: {inlier_ratio:.2f}")
                print(f"  - 使用的融合权重: {blend_weight:.2f}")
            except Exception as e:
                print(f"  - 可视化匹配点时出错: {e}")
                blend_weight = 0.5  # 出错时使用默认值
        else:
            blend_weight = 0.5  # 默认权重
            print(f"  - 使用默认融合权重: {blend_weight:.2f}")
        
        # 使用自适应融合
        self.merged_map = adaptive_warp_and_merge(self.map1, self.map2, H, blend_weight)

        merged_map_msg = self.cv2_to_occupancy_grid(self.merged_map)
        self.merged_map_pub_.publish(merged_map_msg)

    def cv2_to_occupancy_grid(self, image: np.ndarray) -> OccupancyGrid:
        msg = OccupancyGrid()

        image = cv2.bitwise_not(image)
        image[image == 255] = 100

        height, width = image.shape

        # 设置头部信息
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        # 设置地图元数据
        msg.info.resolution = 0.05  # 每像素 0.05 米，可根据实际需要调整
        msg.info.width = width
        msg.info.height = height
        msg.info.origin.position.x = 0.0
        msg.info.origin.position.y = 0.0
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0  # 无旋转

        # 转换为 1D occupancy 数据
        flat_data = []
        for y in range(height):
            for x in range(width):
                pixel = image[height - y - 1, x]  # Y轴翻转
                if pixel == 255:
                    flat_data.append(0)   # Free
                elif pixel == 0:
                    flat_data.append(100)  # Occupied
                else:
                    flat_data.append(-1)  # Unknown

        msg.data = flat_data
        return msg

def main(args=None):
    rclpy.init(args=args)
    node = MapMerge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()