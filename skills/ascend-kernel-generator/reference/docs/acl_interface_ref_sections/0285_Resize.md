### Resize

## 功能
根据 scales 调整输入 Tensor 的大小。

## 输入
- **ONNX 版本为 Opset v10 时**
  - `x`：输入 Tensor，数据类型：uint8、int8、int16、int32、int64、float16、float、double。
  - `scales`：与输入 `x` 的维度相等的数组。
- **ONNX 版本为 Opset v11/v12/v14/v15/v16/v17 时**
  - `x`：输入 Tensor，数据类型：float16、float。
  - `sizes`：输出 Tensor 的 size。

## 输出
`y`：输出尺寸调整后的 Tensor，其尺寸由输入的参数 `scales` 决定。

## 属性
- **ONNX 版本为 Opset v10 时**
  - `mode`：string，插值算法，取值包括 `nearest`、`linear`，默认值为 `nearest`。
- **ONNX 版本为 Opset v11/v12/v14/v15/v16/v17 时**
  - `coordinate_transformation_mode`：string，定义缩放后图像与原图像的坐标转换，取值包括 `align_corners`、`asymmetric`、`tf_half_pixel_for_nn`、`tf_crop_and_resize`、`pytorch_half_pixel`、`half_pixel`，默认值为 `half_pixel`。
  - `cubic_coeff_a`：三次插值系数，数据类型为 float，默认值为 -0.75。
  - `exclude_outside`：超出 tensor 外的权重，数据类型为 int，默认值为 0。
  - `mode`：string，插值算法，取值包括 `nearest`、`linear`、`cubic`，默认值为 `nearest`。

## 约束
- 目前仅支持 `nearest` 和 `linear` 插值方式来处理图片，并且需要修改模型将输入 `scales` 或 `sizes` 由 placeholder 改为 const 类型，可以使用 onnxsimplifier 简化模型。
- 当 `mode` 为 `nearest` 时，仅能采用 `round_prefer_ceil` 实现，不支持其余实现如 `round_prefer_floor`。
- 输入是 5 维时：
  - 当前仅支持线性插值模式，即 `mode=linear`，不支持 `mode=nearest` 和 `mode=cubic`。
  - 在线性插值模式下，当前仅支持 `coordinate_transformation_mode=align_corners` 或 `pytorch_half_pixel` 两种坐标模式，其余模式不支持。

## 支持的 ONNX 版本
Opset v10/v11/v12/v14/v15/v16/v17/v18
