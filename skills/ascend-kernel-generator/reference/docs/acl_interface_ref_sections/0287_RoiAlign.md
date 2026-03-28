### RoiAlign

## 功能
在每个ROI（Region of Interest）区域进行池化处理。

## 输入
- **x**：输入Tensor，4D输入
  - 数据类型：float16、float
  - 数据格式：NCHW
  - Shape：(N, C, H, W)
- **rois**：感兴趣区域
  - 数据类型：float16、float
  - Shape：(num_rois, 4)
- **batch_indices**：batch对应图像的索引
  - 数据类型：int64
  - Shape：(num_rois,)

## 输出
- **y**：输出Tensor
  - 数据类型：与输入x相同
  - Shape：(num_rois, C, output_height, output_width)

## 属性
| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| mode | string | avg | 池化方式 |
| output_height | int | 1 | y的高度 |
| output_width | int | 1 | y的宽度 |
| sampling_ratio | int | 0 | 插值算法采样点数 |
| spatial_scale | float | 1.0 | 相对于输入图像的空间采样率 |
| coordinate_transformation_mode | string | half_pixel | 是否对输入值进行偏移（Opset v16及之后版本支持该属性） |

## 约束
不支持atc工具参数`--precision_mode=must_keep_origin_dtype`时float64的输入。

## 支持的 ONNX 版本
Opset v10/v11/v12/v13/v14/v15/v16/v17/v18
