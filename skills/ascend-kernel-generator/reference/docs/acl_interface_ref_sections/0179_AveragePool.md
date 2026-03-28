### AveragePool

## 功能
对输入 Tensor 进行平均池化操作。

## 输入
- **X**：输入 Tensor，数据类型支持 float16、float，格式为 NCHW，数据格式支持 ND。

## 属性
- **auto_pad**（可选）：支持 `NOTSET`、`SAME_UPPER` 与 `VALID`，默认值为 `NOTSET`。
- **count_include_pad**：int，暂不支持。
- **kernel_shape**（可选）：
  - `kernel_shape[0]`：数据类型 int32，指定沿 H 维度的窗口大小，取值范围为 [1, 32768]，默认为 1。
  - `kernel_shape[1]`：数据类型 int32，指定沿 W 维度的窗口大小，取值范围为 [1, 32768]，默认为 1。
- **strides**（可选）：
  - `strides[0]`：数据类型 int32，指定沿 H 维度的步长，默认为 1。
  - `strides[1]`：数据类型 int32，指定沿 W 维度的步长，默认为 1。
- **pads**（可选）：
  - `pads[0]`：数据类型 int32，指定顶部 padding，默认为 0。
  - `pads[1]`：数据类型 int32，指定底部 padding，默认为 0。
  - `pads[2]`：数据类型 int32，指定左部 padding，默认为 0。
  - `pads[3]`：数据类型 int32，指定右部 padding，默认为 0。
- **ceil_mode**（可选）：数据类型 int32，取值：0（floor 模式），1（ceil 模式），默认为 0。

## 输出
- **Y**：输出 Tensor，数据类型支持 float16、float，格式为 NCHW，数据格式支持 ND。

## 约束与限制
- `strides[0]` 或 `strides[1]` 取值步长大于 63 时，会使用 AI CPU 计算，性能会下降。
- `kernel_shape[0]` 或 `kernel_shape[1]` 取值超过 [1, 255]，或 `kernel_shape[0] * kernel_shape[1] > 256` 时，会使用 AI CPU 计算，导致性能下降。
- `1 <= input_w <= 4096`。
- 当输入 tensor 的 N 是一个质数时，N 应当小于 65535。
- `ceil_mode` 参数仅在 `auto_pad='NOTSET'` 时生效。
- 不支持 atc 工具参数 `--precision_mode=must_keep_origin_dtype` 时 float 类型输入。
- `auto_pad` 属性值支持 `SAME_UPPER` 而不支持 `SAME_LOWER`，原因是 `SAME_LOWER` 和 `SAME_UPPER` 统一使用的 TBE 的 SAME 属性，即 TBE 算子没有根据这个属性区分 pad 的填充位置，可能会带来精度问题。
- 滑窗范围超过输入特征图原始宽高的大小，且 `count_include_pad` 为 False 时，可能导致算子计算中的除数分母为 0，输出结果可能为 0、65504、NaN、INF。该场景不符合该算子的正常业务逻辑，建议修改 `ceil_mode` 或 `stride` 等属性，满足滑窗始终与输入特征图的有交集。

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
