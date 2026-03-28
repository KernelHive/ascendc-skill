### ArgMax

## 功能
返回输入 Tensor 指定轴上最大值对应的索引。

## 输入
- **data**：输入 Tensor，数据类型支持 float16、float、int32，数据格式支持 ND。

## 属性
- **axis**：表示计算最大值索引的方向，数据类型为 int32，取值范围为 `[-len(x.shape), len(x.shape)-1]`。
- **keepdims**：可选属性，默认为 1，支持 1 或 0。
- **select_last_index**：如果最大值出现在多个位置，是否选择第一个位置或最后一个位置，取值范围为 `[0, 1]`，默认为 0，目前只支持 0。

## 输出
- **reduced**：输出 Tensor，表示最大值的索引位置，维度比输入 x 少 1，数据类型支持 int64，数据格式支持 ND。

## 约束与限制
无

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
