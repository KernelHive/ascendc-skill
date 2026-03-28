##### SetOutputAttr

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

设置Graph的输出属性，泛型属性接口，属性类型为attr_value。

## 函数原型

```cpp
graphStatus SetOutputAttr(const AscendString &name, uint32_t output_index, const AttrValue &attr_value)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| name | 输入 | 输出属性名 |
| output_index | 输入 | 输出节点索引 |
| attr_value | 输入 | 输出的属性值 |

## 返回值说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| - | graphStatus | GRAPH_SUCCESS(0)：成功<br>其他值：失败 |

## 约束说明

无
