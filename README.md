# 地图拼接融合项目

这个项目用于将两个机器人观测到的地图进行拼接融合。该脚本能够自动寻找两个地图的重叠部分，通过特征提取和匹配实现地图的对齐和融合。白色表示障碍物，黑色表示空区域。

## 功能特点

- 多层次特征提取和匹配（ORB、边缘、形态学处理）
- 基于几何约束的特征匹配过滤
- 刚性变换计算（旋转和平移，无透视变形）
- 自适应权重的地图融合
- 多数据集批量处理能力
- 详细的可视化输出和日志记录
- 鲁棒的错误处理机制

## 技术实现

本项目使用了以下技术和算法：  

- **多层次特征提取**：同时使用ORB、边缘和形态学处理后的特征，提高特征匹配的成功率
- **几何约束过滤**：使用位移向量一致性、尺度一致性和角度一致性进行过滤
- **自适应回退机制**：当严格过滤导致匹配点不足时，递进式放宽条件
- **刚性变换矩阵计算**：计算仅包含旋转和平移的变换矩阵
- **RANSAC算法**：用于在计算变换矩阵时过滤异常值
- **加权融合策略**：根据匹配质量自适应调整融合权重

## 环境要求

- Python 3.6+
- OpenCV 4.x
- NumPy
- Matplotlib

可以通过以下命令安装依赖：

```bash
pip install opencv-python numpy matplotlib
```

## 使用方法

1. 准备地图数据集，目录结构如下：
```
train_test/
  ├── 数据集1/
  │   ├── map1.png
  │   └── map2.png
  ├── 数据集2/
  │   ├── map1.png
  │   └── map2.png
  └── ...
```

2. 运行脚本：

```bash
python map_merge.py
```

3. 脚本将在`results/exp_时间戳/`目录下生成每个数据集的结果：
   - `filtered_matches.png`：显示经过过滤后的特征点匹配结果
   - `merged.png`：融合后的地图
   - `comparison.png`：原始地图和融合地图的对比图

## 主要函数说明

- `load_map()`：加载地图图像文件
- `enhance_image()`：增强图像以提高特征检测质量
- `detect_features()`：检测图像中的特征点和计算描述子
- `match_features()`：匹配两组特征描述子
- `filter_matches_by_geometry()`：基于几何约束进一步过滤匹配
- `find_rigid_transformation()`：计算两个图像之间的刚性变换矩阵（仅旋转和平移）
- `extract_multi_level_features()`：提取图像的多层次特征
- `match_with_multi_level_features()`：使用多层次特征进行匹配
- `adaptive_warp_and_merge()`：使用自适应权重进行地图融合
- `process_map_pair()`：处理一对地图并保存结果
- `main()`：主函数，协调整个处理流程

## 参数调优

如果匹配效果不理想，可以尝试调整以下参数：

1. `detect_features()`函数中的ORB参数：
   - `nfeatures`：控制检测的特征点数量
   - `scaleFactor`：金字塔层级间的比例因子
   - `nlevels`：金字塔层级数

2. `match_features()`和`relaxed_match_features()`函数中的匹配参数：
   - Lowe比率测试阈值：控制特征匹配的严格程度

3. `filter_matches_by_geometry()`函数中的过滤参数：
   - 位移向量一致性阈值
   - 尺度一致性阈值
   - 角度一致性阈值

4. `find_rigid_transformation()`函数中的参数：
   - `ransacReprojThreshold`：控制RANSAC算法的容差
   - `ratio`参数：控制用于计算变换矩阵的匹配点比例

5. `adaptive_warp_and_merge()`函数中的`blend_weight`参数：
   - 控制两个地图在重叠区域的融合权重

## 改进历史

最新的改进包括：
- 使用`estimateAffinePartial2D`替代`findHomography`进行变换矩阵计算，确保地图只进行旋转和平移变换，不会发生透视变形
- 增加多层次特征提取和匹配策略，提高匹配成功率
- 优化几何约束过滤机制，使用更全面的约束条件
- 添加自适应权重的地图融合方法 