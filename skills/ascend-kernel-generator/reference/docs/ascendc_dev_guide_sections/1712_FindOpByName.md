##### FindOpByName

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

基于算子名称，获取缓存在 Graph 中的 op 对象。

## 函数原型

> **须知**
> 
> 数据类型为 string 的接口后续版本会废弃，建议使用数据类型为非 string 的接口。

```cpp
graphStatus FindOpByName(const std::string &name, ge::Operator &op) const
graphStatus FindOpByName(const char_t *name, ge::Operator &op) const
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| name | 输入 | 需要获取的算子名称。 |
| op | 输出 | 返回用户所需要的 op 对象。 |

## 返回值说明

| 返回值 | 类型 | 说明 |
|--------|------|------|
| - | graphStatus | SUCCESS：成功获取算子实例。<br>FAILED：此名字没有在 Graph 中注册 op 对象。 |

## 约束说明

此接口为非必须接口，与 `15.2.3.9.5 AddOp` 搭配使用。
