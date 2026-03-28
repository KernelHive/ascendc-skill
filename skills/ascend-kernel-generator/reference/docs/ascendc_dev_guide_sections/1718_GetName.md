##### GetName

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

获取当前图的名称。

## 函数原型

> **须知**
> 
> 数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```cpp
const std::string &Graph::GetName() const
graphStatus GetName(AscendString &name) const
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| name | 输出 | 需要获取的图的名称。 |

## 返回值说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| - | graphStatus | SUCCESS：成功获取图的名称。 |

## 约束说明

无
