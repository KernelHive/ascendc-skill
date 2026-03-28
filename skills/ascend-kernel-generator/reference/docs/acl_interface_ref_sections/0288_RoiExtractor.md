### RoiExtractor

## 功能
从特征映射列表中获取 ROI（Region of Interest）特征矩阵。

## 输入
- **features**：输入特征 Tensor，数据类型：float、float16。shape 维度为 4 维。
- **rois**：输入感兴趣区域 Tensor，数据类型：float、float16。

## 输出
- **y**：输出 Tensor，数据类型：float、float16。shape 同输入 features 一致。

## 属性
- **finest_scale**（可选）：数据类型为 int。默认值 56，最精细尺度。
- **roi_scale_factor**（可选）：数据类型为 float。默认值 0，感兴趣区域（ROI）缩放相关的因子。
- **spatial_scale**（可选）：数据类型为 float。空间尺度因子。
- **pooled_height**（可选）：数据类型为 int。默认值 7，池化操作中的高度维度。
- **pooled_width**（可选）：数据类型为 int。默认值 7，池化操作中的宽度维度。
- **sample_num**（可选）：数据类型为 int。默认值 0，采样操作中的样本数量。
- **pool_mode**（可选）：数据类型为 string。池化模式。
- **aligned**（可选）：数据类型为 bool。默认 true，是否对齐操作。

## 约束
无

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16
