### LSTMP

## 功能
计算单层LSTMP。

## 输入
- **x**：输入Tensor，数据类型支持float16
- **wx**：输入Tensor，数据类型支持float16
- **bias**：输入Tensor，数据类型支持float16
- **wr**：输入Tensor，数据类型支持float16
- **project**：输入Tensor，数据类型支持float16
- **real_mask**（可选）：输入Tensor，数据类型支持float16
- **init_h**（可选）：输入Tensor，数据类型支持float16
- **init_c**（可选）：输入Tensor，数据类型支持float16

## 输出
1. **y**：输出Tensor，数据类型：float16
2. **output_h**：输出Tensor，数据类型：float16
3. **output_c**：输出Tensor，数据类型：float16

## 约束与限制
该算子为特定用户场景定制，非相关场景不建议使用。

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16
