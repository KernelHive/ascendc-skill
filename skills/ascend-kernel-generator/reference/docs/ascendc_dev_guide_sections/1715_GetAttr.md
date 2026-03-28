##### GetAttr

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

获取Graph的属性，泛型属性接口，属性的类型为attr_value。

## 函数原型

```cpp
graphStatus GetAttr(const AscendString &name, AttrValue &attr_value) const
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| name | 输出 | 返回Graph的属性名称 |
| attr_value | 输出 | 返回Graph的属性值 |

## 返回值说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| graphStatus | - | GRAPH_SUCCESS(0)：成功<br>其他值：失败 |

## 约束说明

无

## 调用示例

```cpp
AttrValue av_get;
graph.GetAttr("int_attr", av_get);
int64_t int_attr_get{};
av_get.GetAttrValue(int_attr_get);
ASSERT_EQ(int_attr_get, 100);
```
