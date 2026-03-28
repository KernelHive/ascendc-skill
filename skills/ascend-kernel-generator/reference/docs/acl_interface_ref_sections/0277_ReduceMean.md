### ReduceMean

## 功能
计算输入 Tensor 在指定维度上的元素均值。

如果 `keepdims` 等于 1，输出 Tensor 的秩与输入相同；如果 `keepdims` 等于 0，输出 Tensor 的维度将被缩减。

## 输入
- **x**：输入 Tensor，支持 float16、float 数据类型。

## 属性
- **axes**：int 列表类型，指定要缩减的维度。
- **keepdims**：int 类型，默认值为 1，表示是否保留缩减后的维度。

## 输出
- **y**：输出 Tensor，数据类型与输入 `x` 相同。

## 约束与限制
当 `axes` 为空时，当前输出不会执行全维度规约。建议：
- 将 ONNX 算子的 `axes` 设置为所有轴；
- 或在 PyTorch 导出 ONNX 图之前，在 PyTorch 模型中使用 `amax` 对所有轴进行规约（例如：`x.amax(dim=[0, 1, 2])`）。

## 支持的 ONNX 版本
Opset v8、v9、v10、v11、v12、v13、v14、v15、v16。
