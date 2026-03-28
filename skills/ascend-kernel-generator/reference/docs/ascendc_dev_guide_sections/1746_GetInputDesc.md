##### GetInputDesc

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

获取指定输入端口的 tensor 格式。

## 函数原型

```cpp
graphStatus GetInputDesc(const int32_t index, TensorDesc &tensor_desc) const
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| index | 输入 | 输入的 index |
| tensor_desc | 输出 | 获取到的 tensor 格式 |

## 返回值说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| - | graphStatus | GRAPH_SUCCESS(0)：成功<br>其他值：失败 |

## 约束说明

无
