### AffineGrid

## 功能

给定一组 3 维的仿射参数矩阵（theta）以及输出图像的大小（size），生成一个 2D 或 3D 的网格，该网格表示仿射后图像的点在原图像上的坐标。

## 输入

- **theta**：输入 Tensor，shape 为 2D (N, 2, 3) 或 3D (N, 3, 4)，数据类型支持 float16、float，数据格式支持 ND。
- **size**：输入 Tensor，2D 的目标输出图像尺寸 (N, C, H, W) 或 3D 的目标输出图像尺寸 (N, C, D, H, W)，数据类型支持 int32，数据格式支持 ND。

## 属性

- **align_corners**：bool 型，表示是否角像素点对齐。

## 输出

- **grid**：输出 Tensor，数据类型支持 int，数据格式支持 ND。

## 约束与限制

无

## 支持的 ONNX 版本

Opset v8/v9/v10/v11/v12/v13/v14/v15/v16
