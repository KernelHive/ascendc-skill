##### SetAttr

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

设置Graph的属性，泛型属性接口，属性的类型为attr_value。

## 函数原型

```cpp
graphStatus SetAttr(const AscendString &name, const AttrValue &attr_value)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| name | 输入 | 属性名称 |
| attr_value | 输入 | 属性值 |

## 返回值说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| graphStatus | - | GRAPH_SUCCESS(0)：成功<br>其他值：失败 |

## 约束说明

无

## 调用示例

```cpp
AttrValue av;
av.SetAttrValue(static_cast<int64_t>(100));
Graph graph("test");
graph.SetAttr("int_attr", av);
```
