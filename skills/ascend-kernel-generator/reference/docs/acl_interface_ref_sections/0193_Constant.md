### Constant

## 功能
构建一个常量 Tensor。

## 属性
- **value**（必选）：输出 Tensor 的元素值。  
  支持的数据类型：float、float16、bfloat16、int4、int8、int16、int32、int64、uint8、uint16、uint32、uint64、bool、double。

## 输出
- **y**：输出 Tensor，包含与提供的张量相同的值。  
  支持的数据类型：float、float16、bfloat16、int4、int8、int16、int32、int64、uint8、uint16、uint32、uint64、bool、double。

## 约束
- `value` 不支持稀疏格式。

## 支持的 ONNX 版本
Opset v8、v9、v10、v11、v12、v13
