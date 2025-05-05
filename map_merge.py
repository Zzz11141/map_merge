#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
地图拼接融合脚本 - 改进版

此脚本用于将两个机器人观测到的地图进行拼接融合。
其中白色表示障碍物，黑色表示空区域。脚本会自动寻找
两个地图的重叠部分，并进行旋转和平移操作以实现拼接。
改进版使用更精确的特征点匹配算法。
"""

import cv2 
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import os
import glob
import traceback
import datetime

def load_map(file_path: str) -> np.ndarray:
    """
    加载地图图像文件

    参数:
        file_path: 图像文件路径

    返回:
        加载的图像数组
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
    
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {file_path}")
    
    return img

def enhance_image(img: np.ndarray) -> np.ndarray:
    """
    增强图像以提高特征检测质量
    
    参数:
        img: 输入图像
        
    返回:
        增强后的图像
    """
    # 应用CLAHE（对比度受限的自适应直方图均衡化）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    
    # 应用高斯模糊去除噪声
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    return enhanced

def detect_features(img: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """
    检测图像中的特征点和计算描述子

    参数:
        img: 输入图像

    返回:
        特征点列表和对应的描述子
    """
    # 图像增强
    enhanced_img = enhance_image(img)
    
    # 使用ORB特征检测器 (Oriented FAST and Rotated BRIEF)
    orb = cv2.ORB_create(
        nfeatures=3000,       # 增加特征点数量
        scaleFactor=1.2,      # 金字塔层级间的比例因子
        nlevels=8,            # 金字塔层级数
        edgeThreshold=31,     # 边缘阈值
        firstLevel=0,
        WTA_K=2,              # 用于计算BRIEF描述子的点数
        patchSize=31          # 特征点附近的区域大小
    )
    
    # 检测关键点和计算描述子
    keypoints, descriptors = orb.detectAndCompute(enhanced_img, None)
    
    # 确保检测到了特征点和描述子
    if keypoints is None or len(keypoints) == 0:
        print(f"  - 警告: 未检测到任何特征点")
        return [], None
        
    if descriptors is None or descriptors.shape[0] == 0:
        print(f"  - 警告: 未能计算特征描述子")
        return keypoints, None
    
    return keypoints, descriptors

def match_features(desc1: np.ndarray, desc2: np.ndarray) -> List[cv2.DMatch]:
    """
    匹配两组特征描述子，使用KNN匹配和Lowe's比率测试
    
    参数:
        desc1: 第一张图像的特征描述子
        desc2: 第二张图像的特征描述子
        
    返回:
        特征匹配结果列表
    """
    # 创建BFMatcher对象
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    # 使用KNN找到每个描述子的两个最佳匹配
    raw_matches = matcher.knnMatch(desc1, desc2, k=2)
    
    # 应用Lowe's比率测试筛选良好匹配
    good_matches = []
    for matches in raw_matches:
        # 有时候只能找到一个匹配
        if len(matches) < 2:
            continue
            
        m, n = matches
        # 确保最佳匹配的距离小于次佳匹配的距离的一定比例
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    # 按照距离排序
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    
    # 进一步筛选距离较近的匹配
    if len(good_matches) > 10:
        # 计算平均距离和标准差
        distances = np.array([m.distance for m in good_matches])
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        # 筛选距离小于(平均距离 + 1.5*标准差)的匹配
        filtered_matches = [m for m in good_matches 
                           if m.distance < mean_dist + 1.5 * std_dist]
        
        if len(filtered_matches) >= 10:
            print(f"  - 经距离过滤后的匹配点数量: {len(filtered_matches)}")
            return filtered_matches
    
    return good_matches

def filter_matches_by_geometry(kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint], 
                              matches: List[cv2.DMatch]) -> List[cv2.DMatch]:
    """
    基于几何约束进一步过滤匹配
    
    参数:
        kp1: 第一张图像的关键点
        kp2: 第二张图像的关键点
        matches: 初步匹配结果
        
    返回:
        过滤后的匹配结果
    """
    if len(matches) < 10:
        return matches
    
    # 获取匹配点坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    # 计算匹配点的尺度和方向
    scales1 = np.float32([kp1[m.queryIdx].size for m in matches])
    scales2 = np.float32([kp2[m.trainIdx].size for m in matches])
    
    angles1 = np.float32([kp1[m.queryIdx].angle for m in matches])
    angles2 = np.float32([kp2[m.trainIdx].angle for m in matches])
    
    # 计算尺度比率和角度差异
    scale_ratios = scales1 / (scales2 + 1e-6)  # 防止除零
    angle_diffs = np.abs(angles1 - angles2)
    angle_diffs = np.minimum(angle_diffs, 360 - angle_diffs)  # 角度差在[0, 180]之间
    
    # 计算各点之间的相对位移
    displacement_vectors = dst_pts - src_pts
    
    # 计算位移向量的平均值和标准差
    mean_vector = np.mean(displacement_vectors, axis=0)
    std_vector = np.std(displacement_vectors, axis=0)
    
    # 计算位移向量与平均向量的偏差
    deviations = np.sqrt(np.sum((displacement_vectors - mean_vector)**2, axis=1))
    mean_dev = np.mean(deviations)
    std_dev = np.std(deviations)
    
    # 计算尺度比率和角度差异的统计特性
    mean_scale_ratio = np.mean(scale_ratios)
    std_scale_ratio = np.std(scale_ratios)
    
    mean_angle_diff = np.mean(angle_diffs)
    std_angle_diff = np.std(angle_diffs)
    
    print(f"  - 初始匹配点数: {len(matches)}")
    print(f"  - 位移向量标准差: X={std_vector[0]:.2f}, Y={std_vector[1]:.2f}")
    
    # 组合多个几何约束条件进行筛选
    geometric_matches = []
    for i in range(len(matches)):
        # 位移向量一致性检查 - 欧氏距离
        is_displacement_ok = deviations[i] < mean_dev + 2.0 * std_dev
        
        # 位移向量一致性检查 - 分量检查
        # 充分利用std_vector对X和Y方向上的偏差进行约束
        x_deviation = abs(displacement_vectors[i, 0] - mean_vector[0])
        y_deviation = abs(displacement_vectors[i, 1] - mean_vector[1])
        is_vector_component_ok = (x_deviation < 2.5 * std_vector[0]) and (y_deviation < 2.5 * std_vector[1])
        
        # 尺度一致性检查
        is_scale_ok = abs(scale_ratios[i] - mean_scale_ratio) < 2.0 * std_scale_ratio
        
        # 角度一致性检查
        is_angle_ok = abs(angle_diffs[i] - mean_angle_diff) < 2.0 * std_angle_diff
        
        # 组合所有约束条件
        if is_displacement_ok and is_vector_component_ok and is_scale_ok and is_angle_ok:
            geometric_matches.append(matches[i])
    
    print(f"  - 经几何约束过滤后的匹配点数量: {len(geometric_matches)}")
    
    # 以不同级别的宽松度进行三级回退策略
    if len(geometric_matches) < 10:
        print(f"  - 几何约束太严格，尝试第一级回退")
        geometric_matches = []
        for i in range(len(matches)):
            # 放宽位移向量和分量检查
            is_displacement_ok = deviations[i] < mean_dev + 2.5 * std_dev
            x_deviation = abs(displacement_vectors[i, 0] - mean_vector[0])
            y_deviation = abs(displacement_vectors[i, 1] - mean_vector[1])
            is_vector_component_ok = (x_deviation < 3.0 * std_vector[0]) and (y_deviation < 3.0 * std_vector[1])
            
            # 保持尺度和角度约束
            is_scale_ok = abs(scale_ratios[i] - mean_scale_ratio) < 2.0 * std_scale_ratio
            is_angle_ok = abs(angle_diffs[i] - mean_angle_diff) < 2.0 * std_angle_diff
            
            if is_displacement_ok and is_vector_component_ok and (is_scale_ok or is_angle_ok):  # 放宽逻辑运算
                geometric_matches.append(matches[i])
                
        print(f"  - 第一级回退后的匹配点数量: {len(geometric_matches)}")
        
        # 如果仍然太少，进一步放宽
        if len(geometric_matches) < 8:
            print(f"  - 第一级回退仍然太严格，尝试第二级回退")
            geometric_matches = []
            for i in range(len(matches)):
                # 进一步放宽位移检查，仍然使用std_vector
                is_displacement_ok = deviations[i] < mean_dev + 3.0 * std_dev
                x_deviation = abs(displacement_vectors[i, 0] - mean_vector[0])
                y_deviation = abs(displacement_vectors[i, 1] - mean_vector[1])
                is_vector_component_ok = (x_deviation < 3.5 * std_vector[0]) and (y_deviation < 3.5 * std_vector[1])
                
                # 只检查位移，不检查尺度和角度
                if is_displacement_ok and is_vector_component_ok:
                    geometric_matches.append(matches[i])
                    
            print(f"  - 第二级回退后的匹配点数量: {len(geometric_matches)}")
            
            # 最后的回退策略
            if len(geometric_matches) < 6:
                print(f"  - 第二级回退仍然太严格，使用最终回退")
                geometric_matches = [matches[i] for i in range(len(matches)) 
                                    if deviations[i] < mean_dev + 3.5 * std_dev]
                print(f"  - 最终匹配点数量: {len(geometric_matches)}")
    
    return geometric_matches

def find_rigid_transformation(kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint], 
                             matches: List[cv2.DMatch], ratio: float = 0.8) -> Tuple[Optional[np.ndarray], Optional[List[cv2.DMatch]], Optional[np.ndarray]]:
    """
    计算两个图像之间的刚性变换矩阵（仅旋转和平移）
    
    参数:
        kp1: 第一张图像的关键点
        kp2: 第二张图像的关键点
        matches: 特征匹配结果
        ratio: 匹配筛选比例
        
    返回:
        (变换矩阵, 过滤后的匹配点, RANSAC内点掩码)
    """
    if len(matches) < 10:
        print("  - 找不到足够的匹配点（至少需要10个）")
        return None, None, None
        
    # 应用几何约束过滤
    filtered_matches = filter_matches_by_geometry(kp1, kp2, matches)
    
    if len(filtered_matches) < 10:
        print("  - 几何过滤后的匹配点不足，使用原始匹配点")
        filtered_matches = matches
        
    # 选择最佳匹配
    good_matches = filtered_matches[:int(len(filtered_matches) * ratio)]
    
    if len(good_matches) < 4:
        print("  - 筛选后的匹配点不足4个")
        return None, None, None
    
    try:
        # 获取匹配点的坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
        # 使用RANSAC计算刚性变换矩阵（旋转+平移）
        partial_affine, inliers = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
        
        if partial_affine is None or inliers is None:
            print("  - 无法计算刚性变换矩阵")
            return None, None, None
            
        # 将2x3仿射矩阵转换为3x3的变换矩阵
        H = np.eye(3, dtype=np.float32)
        H[:2, :] = partial_affine
            
        # 计算内点比例
        inlier_ratio = np.sum(inliers) / len(inliers)
        print(f"  - RANSAC内点比例: {inlier_ratio:.2f}")
        
        # 如果内点比例太低，可能是错误匹配
        if inlier_ratio < 0.3:
            print("  - 内点比例过低，变换可能不可靠，使用更宽松的阈值重新计算")
            partial_affine, inliers = cv2.estimateAffinePartial2D(
                src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0
            )
            
            if partial_affine is None or inliers is None:
                print("  - 重新计算失败")
                return None, None, None
                
            H = np.eye(3, dtype=np.float32)
            H[:2, :] = partial_affine
            
            inlier_ratio = np.sum(inliers) / len(inliers)
            print(f"  - 重新计算后的内点比例: {inlier_ratio:.2f}")
            
            if inlier_ratio < 0.2:
                print("  - 内点比例仍然过低，拒绝变换")
                return None, None, None
                
        return H, good_matches, inliers
    except Exception as e:
        print(f"  - 计算变换矩阵时出错: {e}")
        return None, None, None

def warp_and_merge(img1: np.ndarray, img2: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    根据变换矩阵将第一张图像变换并与第二张图像融合（仅旋转和平移，无缩放）

    参数:
        img1: 第一张图像
        img2: 第二张图像
        H: 变换矩阵

    返回:
        融合后的图像
    """
    # 获取图像尺寸
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    
    # 获取仿射变换矩阵（只取前两行，保证只有旋转和平移）
    affine_matrix = H[:2, :]
    
    # 计算变换后的图像边界点
    corners = np.array([
        [0, 0],
        [w1-1, 0],
        [w1-1, h1-1],
        [0, h1-1]
    ], dtype=np.float32)
    
    # 变换四个角点来确定新图像的尺寸
    transformed_corners = cv2.transform(corners.reshape(1, -1, 2), affine_matrix)
    transformed_corners = transformed_corners.reshape(-1, 2)
    
    # 计算包含所有点的边界框
    min_x = np.min(transformed_corners[:, 0])
    max_x = np.max(transformed_corners[:, 0])
    min_y = np.min(transformed_corners[:, 1])
    max_y = np.max(transformed_corners[:, 1])
    
    # 计算偏移量
    offset_x = max(0, -int(min_x))
    offset_y = max(0, -int(min_y))
    
    # 计算输出图像的尺寸
    output_width = max(int(max_x) + offset_x, w2 + offset_x)
    output_height = max(int(max_y) + offset_y, h2 + offset_y)
    
    # 创建包含偏移的变换矩阵
    warp_matrix = np.copy(affine_matrix)
    warp_matrix[0, 2] += offset_x
    warp_matrix[1, 2] += offset_y
    
    # 创建输出图像
    merged_img = np.zeros((output_height, output_width), dtype=np.uint8)
    
    # 将第二张图像放置到画布上
    merged_img[offset_y:offset_y + h2, offset_x:offset_x + w2] = img2
    
    # 变换第一张图像 - 使用warpAffine确保只有旋转和平移，没有缩放
    warped_img1 = np.zeros((output_height, output_width), dtype=np.uint8)
    cv2.warpAffine(
        img1, warp_matrix, (output_width, output_height),
        dst=warped_img1,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    # 融合两张图像 - 二进制逻辑
    mask = (warped_img1 > 0)
    merged_img[mask] = warped_img1[mask]
    
    print("  - 使用仿射变换(仅旋转和平移)进行地图融合，没有缩放")
    
    return merged_img

def generate_comparison_image(map1: np.ndarray, map2: np.ndarray, merged_map: np.ndarray, 
                             folder_name: str, output_path: str) -> None:
    """
    生成三张地图的对比图像

    参数:
        map1: 第一张地图
        map2: 第二张地图
        merged_map: 融合后的地图
        folder_name: 数据集文件夹名
        output_path: 输出路径
    """
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(map1, cmap='gray')
    plt.title("Map 1 (Moving Map)")
    plt.axis('on')
    
    plt.subplot(132)
    plt.imshow(map2, cmap='gray')
    plt.title("Map 2 (Fixed Map)")
    plt.axis('on')
    
    plt.subplot(133)
    plt.imshow(merged_map, cmap='gray')
    plt.title("Merged Map")
    plt.axis('on')
    
    plt.suptitle(f"Map Merging Results - {folder_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def visualize_matches(img1: np.ndarray, kp1: List[cv2.KeyPoint], 
                     img2: np.ndarray, kp2: List[cv2.KeyPoint], 
                     matches: List[cv2.DMatch], mask: np.ndarray = None,
                     title: str = "Feature Matches", 
                     output_path: str = None) -> np.ndarray:
    """
    可视化特征点匹配结果
    
    参数:
        img1: 第一张图像
        kp1: 第一张图像的关键点
        img2: 第二张图像
        kp2: 第二张图像的关键点
        matches: 特征匹配结果
        mask: RANSAC掩码，标识内点
        title: 图像标题
        output_path: 输出路径
        
    返回:
        可视化结果图像
    """
    # 限制匹配点数量
    max_matches = min(50, len(matches))
    matches_to_draw = matches[:max_matches]
    
    # 准备绘制参数
    draw_params = dict(
        matchColor=(0, 255, 0),     # 匹配点连线为绿色
        singlePointColor=None,
        flags=2                     # 绘制匹配点和连线
    )
    
    # 如果有掩码，只显示内点，但需要确保掩码大小与匹配点数量一致
    if mask is not None:
        # 截取与matches_to_draw相同数量的掩码
        mask_to_use = mask[:max_matches]
        draw_params['matchesMask'] = mask_to_use.ravel().tolist()
        
    # 绘制匹配结果
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, 
                               matches_to_draw, None, **draw_params)
    
    # 保存结果
    if output_path:
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    return match_img

def extract_multi_level_features(img: np.ndarray) -> dict:
    """
    提取图像的多层次特征，但只使用增强的ORB特征
    
    参数:
        img: 输入图像
        
    返回:
        包含ORB特征的字典
    """
    features = {}
    
    # 图像增强
    enhanced_img = enhance_image(img)
    
    # 结构增强处理 - 增加边缘检测以提高特征点质量
    structure_img = cv2.Canny(enhanced_img, 40, 120)
    kernel = np.ones((3, 3), np.uint8)
    structure_img = cv2.dilate(structure_img, kernel, iterations=1)
    structure_img = cv2.add(enhanced_img, cv2.cvtColor(structure_img, cv2.COLOR_GRAY2BGR)[:,:,0])
    
    # 创建ORB特征检测器 - 使用更宽松的参数
    orb = cv2.ORB_create(
        nfeatures=5000,        # 大幅增加特征点数量
        scaleFactor=1.1,       # 更小的比例因子，检测更多尺度
        nlevels=10,            # 增加金字塔层级数
        edgeThreshold=20,      # 降低边缘阈值，检测更多边缘特征
        firstLevel=0,
        WTA_K=2,               # 保持默认BRIEF描述子设置
        patchSize=31,          # 特征点附近的区域大小
        fastThreshold=15       # 降低FAST检测器阈值，检测更多角点
    )
    
    # 在增强图像上检测特征
    kp_orb, desc_orb = orb.detectAndCompute(enhanced_img, None)
    
    # 在结构增强图像上检测特征
    kp_structure, desc_structure = orb.detectAndCompute(structure_img, None)
    
    # 合并两组ORB特征点 (如果都有检测到的话)
    if (kp_orb is not None and len(kp_orb) > 0 and desc_orb is not None and 
        kp_structure is not None and len(kp_structure) > 0 and desc_structure is not None):
        # 将两组特征点和描述子合并
        kp_combined = kp_orb + kp_structure
        desc_combined = np.vstack((desc_orb, desc_structure))
        features['orb'] = (kp_combined, desc_combined)
    elif kp_structure is not None and len(kp_structure) > 0 and desc_structure is not None:
        features['orb'] = (kp_structure, desc_structure)
    else:
        features['orb'] = (kp_orb, desc_orb)
    
    return features

def match_with_multi_level_features(img1: np.ndarray, img2: np.ndarray) -> Tuple[Optional[np.ndarray], List[cv2.KeyPoint], List[cv2.KeyPoint], List[cv2.DMatch], str, Optional[List[cv2.DMatch]], Optional[np.ndarray]]:
    """
    只使用ORB特征进行匹配，并使用更宽松的参数
    
    参数:
        img1: 第一张图像
        img2: 第二张图像
        
    返回:
        (变换矩阵, 特征点1, 特征点2, 原始匹配结果, 使用的特征类型, 过滤后的匹配点, RANSAC内点掩码)
    """
    # 提取ORB特征
    features1 = extract_multi_level_features(img1)
    features2 = extract_multi_level_features(img2)
    
    # 获取ORB特征
    kp1, desc1 = features1['orb']
    kp2, desc2 = features2['orb']
    
    # 检查特征点是否有效
    if desc1 is None or desc2 is None or len(kp1) < 4 or len(kp2) < 4:
        print(f"  - ORB特征无效，无法进行匹配")
        return None, None, None, None, None, None, None
            
    print(f"  - 使用ORB特征进行匹配")
    print(f"  - ORB特征点数量: {len(kp1)}/{len(kp2)}")
    
    # 使用宽松的参数进行匹配
    try:
        # 1. 首先尝试KNN匹配 + 更宽松的Lowe比率测试 (0.7 -> 0.85)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw_matches = matcher.knnMatch(desc1, desc2, k=2)
        
        good_matches = []
        for matches in raw_matches:
            if len(matches) < 2:
                # 如果只有一个匹配，直接添加
                good_matches.append(matches[0])
                continue
                
            m, n = matches
            # 使用更宽松的比率
            if m.distance < 0.85 * n.distance:
                good_matches.append(m)
        
        # 按距离排序
        matches = sorted(good_matches, key=lambda x: x.distance)
        print(f"  - KNN匹配后点数: {len(matches)}")
        
        # 如果匹配点太少，尝试直接匹配
        if len(matches) < 10:
            print(f"  - KNN匹配点不足，尝试直接匹配")
            matches = matcher.match(desc1, desc2)
            # 按距离排序并保留70%的最佳匹配
            matches = sorted(matches, key=lambda x: x.distance)[:int(len(matches) * 0.7)]
            print(f"  - 直接匹配后点数: {len(matches)}")
            
        # 如果找到足够的匹配点，计算变换矩阵（使用更宽松的比例）
        if len(matches) >= 4:
            # 使用更宽松的参数
            H, filtered_matches, inliers = find_rigid_transformation(kp1, kp2, matches, ratio=0.95)
            
            if H is not None:
                print(f"  - 使用ORB特征找到变换矩阵")
                return H, kp1, kp2, matches, 'orb', filtered_matches, inliers
                
        # 如果还是无法找到足够的内点，尝试再次放宽参数
        if len(matches) >= 4:
            print(f"  - 尝试使用更宽松的RANSAC参数")
            # 进一步放宽RANSAC参数，修改find_rigid_transformation中的默认阈值
            try:
                # 获取匹配点的坐标
                src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
                
                # 使用RANSAC计算刚性变换矩阵（旋转+平移），但使用更宽松的阈值
                partial_affine, inliers = cv2.estimateAffinePartial2D(
                    src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=8.0
                )
                
                if partial_affine is not None and inliers is not None:
                    # 将2x3仿射矩阵转换为3x3的变换矩阵
                    H = np.eye(3, dtype=np.float32)
                    H[:2, :] = partial_affine
                        
                    # 计算内点比例
                    inlier_ratio = np.sum(inliers) / len(inliers)
                    print(f"  - 宽松RANSAC内点比例: {inlier_ratio:.2f}")
                    
                    # 只要有内点，就接受变换
                    if inlier_ratio > 0.05:  # 极低的内点要求
                        print(f"  - 使用宽松参数找到变换矩阵")
                        # 选择good_matches与inliers的部分
                        filtered_matches = [matches[i] for i in range(len(matches)) if inliers[i]]
                        return H, kp1, kp2, matches, 'orb', filtered_matches, inliers
            except Exception as e:
                print(f"  - 计算宽松变换矩阵时出错: {e}")
                
    except Exception as e:
        print(f"  - ORB特征匹配出错: {e}")
    
    print("  - 无法找到有效变换矩阵")
    return None, None, None, None, None, None, None

def relaxed_match_features(desc1: np.ndarray, desc2: np.ndarray, ratio: float = 0.8) -> List[cv2.DMatch]:
    """
    使用更宽松的参数进行特征匹配
    
    参数:
        desc1: 第一张图像的特征描述子
        desc2: 第二张图像的特征描述子
        ratio: Lowe比率测试阈值，越大越宽松
        
    返回:
        特征匹配结果列表
    """
    # 创建BFMatcher对象
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    # 使用KNN找到每个描述子的两个最佳匹配
    raw_matches = matcher.knnMatch(desc1, desc2, k=2)
    
    # 应用更宽松的Lowe比率测试
    good_matches = []
    for matches in raw_matches:
        # 有时候只能找到一个匹配
        if len(matches) < 2:
            continue
            
        m, n = matches
        # 使用更宽松的比率
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    
    # 按照距离排序
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    
    return good_matches

def adaptive_warp_and_merge(img1: np.ndarray, img2: np.ndarray, H: np.ndarray, blend_weight: float = 0.5) -> np.ndarray:
    """
    使用自适应权重进行地图融合（仅旋转和平移，无缩放）
    
    参数:
        img1: 第一张图像
        img2: 第二张图像
        H: 变换矩阵
        blend_weight: 混合权重

    返回:
        融合后的图像
    """
    # 获取图像尺寸
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    
    # 获取仿射变换矩阵（只取前两行，保证只有旋转和平移）
    affine_matrix = H[:2, :]
    
    # 计算变换后的图像边界点
    corners = np.array([
        [0, 0],
        [w1-1, 0],
        [w1-1, h1-1],
        [0, h1-1]
    ], dtype=np.float32)
    
    # 变换四个角点来确定新图像的尺寸
    transformed_corners = cv2.transform(corners.reshape(1, -1, 2), affine_matrix)
    transformed_corners = transformed_corners.reshape(-1, 2)
    
    # 计算包含所有点的边界框
    min_x = np.min(transformed_corners[:, 0])
    max_x = np.max(transformed_corners[:, 0])
    min_y = np.min(transformed_corners[:, 1])
    max_y = np.max(transformed_corners[:, 1])
    
    # 计算偏移量
    offset_x = max(0, -int(min_x))
    offset_y = max(0, -int(min_y))
    
    # 计算输出图像的尺寸
    output_width = max(int(max_x) + offset_x, w2 + offset_x)
    output_height = max(int(max_y) + offset_y, h2 + offset_y)
    
    # 创建包含偏移的变换矩阵
    warp_matrix = np.copy(affine_matrix)
    warp_matrix[0, 2] += offset_x
    warp_matrix[1, 2] += offset_y
    
    # 创建输出图像
    merged_img = np.zeros((output_height, output_width), dtype=np.uint8)
    
    # 将第二张图像放置到画布上
    merged_img[offset_y:offset_y + h2, offset_x:offset_x + w2] = img2
    
    # 变换第一张图像 - 使用warpAffine确保只有旋转和平移，没有缩放
    warped_img1 = np.zeros((output_height, output_width), dtype=np.uint8)
    cv2.warpAffine(
        img1, warp_matrix, (output_width, output_height),
        dst=warped_img1,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    # 融合两张图像
    # 使用加权融合
    overlap = (warped_img1 > 0) & (merged_img > 0)
    
    # 转换为浮点数进行混合
    merged_float = merged_img.astype(np.float32)
    warped_float = warped_img1.astype(np.float32)
    
    # 在重叠区域进行加权平均
    merged_float[overlap] = (1 - blend_weight) * merged_float[overlap] + blend_weight * warped_float[overlap]
    
    # 非重叠区域保持原值
    non_overlap1 = (warped_img1 > 0) & (merged_img == 0)
    merged_float[non_overlap1] = warped_float[non_overlap1]
    
    print("  - 使用仿射变换(仅旋转和平移)进行地图融合，没有缩放")
    
    return merged_float.astype(np.uint8)

def process_map_pair(map_dir: str, exp_output_dir: str) -> None:
    """
    处理一对地图并保存结果 - 使用多层次特征融合策略

    参数:
        map_dir: 包含地图对的目录
        exp_output_dir: 实验输出的目录
    """
    try:
        # 获取文件夹名（用于结果命名）
        folder_name = os.path.basename(map_dir)
        
        # 创建实验文件夹下的文件夹
        folder_output_dir = os.path.join(exp_output_dir, folder_name)
        os.makedirs(folder_output_dir, exist_ok=True)
        
        # 查找地图文件
        map_files = glob.glob(os.path.join(map_dir, "*.png"))
        if len(map_files) < 2:
            print(f"目录 {map_dir} 中没有找到足够的地图文件 (至少需要两个)")
            return
        
        # 确保文件顺序稳定（按文件名排序）
        map_files.sort()
        
        # 加载地图
        map1 = load_map(map_files[0])
        map2 = load_map(map_files[1])
        
        print(f"处理 {folder_name}:")
        print(f"  - 地图1: {os.path.basename(map_files[0])}, 尺寸: {map1.shape}")
        print(f"  - 地图2: {os.path.basename(map_files[1])}, 尺寸: {map2.shape}")
        
        # 使用多层次特征匹配，并获取过滤后的匹配点和内点掩码
        H, kp1, kp2, matches, feature_type, filtered_matches, inliers = match_with_multi_level_features(map1, map2)
        
        if H is not None:
            # 检查并确保变换矩阵只包含旋转和平移
            print(f"  - 变换矩阵已计算 (使用{feature_type}特征)")
            print(f"  - 变换类型: 刚性变换(仅旋转和平移，无缩放)")
            # 提取旋转角度
            rotation_matrix = H[:2, :2]
            angle_rad = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            angle_deg = np.degrees(angle_rad)
            print(f"  - 旋转角度: {angle_deg:.2f}度")
            print(f"  - 平移量: X={H[0, 2]:.2f}, Y={H[1, 2]:.2f}")
        
        if H is None:
            print(f"  - 无法找到有效的变换矩阵，请检查图像是否有足够的共同特征")
            
            # 仍然保存一个简单的组合图像作为结果
            combined_img = np.zeros((max(map1.shape[0], map2.shape[0]), 
                                    map1.shape[1] + map2.shape[1]), dtype=np.uint8)
            combined_img[:map1.shape[0], :map1.shape[1]] = map1
            combined_img[:map2.shape[0], map1.shape[1]:map1.shape[1]+map2.shape[1]] = map2
            
            # 保存结果
            merged_output_path = os.path.join(folder_output_dir, f"merged.png")
            plt.figure(figsize=(10, 8))
            plt.imshow(combined_img, cmap='gray')
            plt.title(f"Combined Maps (No Matching) - {folder_name}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(merged_output_path)
            plt.close()
            
            # 生成对比图像
            comparison_path = os.path.join(folder_output_dir, f"comparison.png")
            generate_comparison_image(map1, map2, combined_img, folder_name, comparison_path)
            
            print(f"  - 无法融合地图，已保存简单组合图像到 {merged_output_path}")
            print(f"  - 对比图像已保存到 {comparison_path}")
            return
        
        print(f"  - 变换矩阵已计算 (使用{feature_type}特征)")
        
        # 可视化匹配结果
        if filtered_matches is not None and len(filtered_matches) > 0:
            print(f"  - 用于验证的匹配点数量: {len(filtered_matches)}")
            
            # 使用已经计算好的内点掩码直接进行可视化
            filtered_matches_path = os.path.join(folder_output_dir, f"filtered_matches.png")
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
                
                # 可视化匹配结果
                visualize_matches(
                    map1, kp1, map2, kp2, filtered_matches, vis_mask,
                    title=f"Filtered {feature_type.upper()} Feature Matches - {folder_name}",
                    output_path=filtered_matches_path
                )
                
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
        merged_map = adaptive_warp_and_merge(map1, map2, H, blend_weight)
        
        # 保存结果
        merged_output_path = os.path.join(folder_output_dir, f"merged.png")
        plt.figure(figsize=(10, 8))
        plt.imshow(merged_map, cmap='gray')
        plt.title(f"Merged Map ({feature_type.upper()}) - {folder_name}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(merged_output_path)
        plt.close()
        
        # 生成对比图像
        comparison_path = os.path.join(folder_output_dir, f"comparison.png")
        generate_comparison_image(map1, map2, merged_map, folder_name, comparison_path)
        
        print(f"  - 地图融合完成 (融合权重: {blend_weight:.2f})")
        print(f"  - 结果已保存到 {merged_output_path}")
        print(f"  - 对比图像已保存到 {comparison_path}")
    
    except Exception as e:
        print(f"  - 处理 {os.path.basename(map_dir)} 时发生错误:")
        print(traceback.format_exc())

def main():
    """主函数"""
    try:
        # 定义数据集和结果目录
        dataset_dir = "train_test"
        results_base_dir = "results"
        
        # 创建带时间戳的实验目录
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = f"exp_{timestamp}"
        exp_output_dir = os.path.join(results_base_dir, exp_dir)
        
        # 确保输出目录存在
        os.makedirs(exp_output_dir, exist_ok=True)
        
        print(f"实验结果将保存到: {exp_output_dir}")
        print(f"使用改进的特征匹配算法")
        
        # 获取所有数据集子目录
        subdirs = [d for d in glob.glob(os.path.join(dataset_dir, "*")) if os.path.isdir(d)]
        
        if not subdirs:
            print(f"在 {dataset_dir} 中没有找到子目录")
            return
        
        print(f"找到 {len(subdirs)} 个子目录进行处理")
        
        # 处理每个子目录
        for subdir in subdirs:
            process_map_pair(subdir, exp_output_dir)
            
        print(f"所有地图对融合完成！结果保存在 {exp_output_dir}")
        
    except Exception as e:
        print(f"发生错误: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
