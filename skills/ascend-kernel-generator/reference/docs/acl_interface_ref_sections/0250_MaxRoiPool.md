### MaxRoiPool

## 功能

ROI最大池消耗一个输入Tensor X和感兴趣区域（ROI），以便在每个ROI上应用最大池，从而产生输出的4-D形状Tensor（num_roi, channels, pooled_shape[0], pooled_shape[1]）。

## 输入

- **x**：输入Tensor，数据类型支持float16
- **rois**：输入Tensor，数据类型支持float16

## 属性

- **pooled_shape**：数据类型为list of ints
- **spatial_scale**：数据类型为float，默认值：1.0

## 输出

**y**：输出Tensor，数据类型：float16。shape为4-D形状Tensor（num_roi, channels, pooled_shape[0], pooled_shape[1]）。

## 约束与限制

不支持atc工具参数`--precision_mode=must_keep_origin_dtype`时float类型输入。

## 支持的 ONNX 版本

Opset v8/v9/v10/v11/v12/v13/v14/v15/v16
