### ArgMin

## 功能
返回输入 Tensor 指定轴上最小值对应的索引。

## 输入
- **data**：输入 Tensor，数据类型支持 float16、float，数据格式支持 ND。

## 属性
- **axis**：数据类型为 int，含义：指定计算轴；取值范围：[-r, r-1]，r 表示输入数据的秩。
- **keepdims**：默认为 1，支持 1 或 0（可选）。
- **select_last_index**：如果最小值出现在多个位置，是否选择第一个位置或最后一个位置，取值范围：[0, 1]，默认为 0，目前只支持 0。

## 输出
- **reduced**：输出 Tensor，数据类型支持 int32，数据格式支持 ND。

## 约束与限制
输入 data 不支持 reduce 轴为 0 的场景。

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
