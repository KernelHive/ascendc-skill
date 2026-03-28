##### GetAllOpName

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

获取 Graph 中已注册的所有缓存算子的名称列表。

## 函数原型

> **须知**
> 数据类型为 string 的接口后续版本会废弃，建议使用数据类型为非 string 的接口。

```cpp
graphStatus GetAllOpName(std::vector<std::string> &op_name) const
graphStatus GetAllOpName(std::vector<AscendString> &names) const
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| op_name | 输出 | 返回 Graph 中的所有算子的名称。 |
| names | 输出 | 返回 Graph 中的所有算子的名称。 |

## 返回值说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| - | graphStatus | SUCCESS：操作成功<br>FAILED：操作失败 |

## 约束说明

此接口为非必需接口，与 15.2.3.9.5 AddOp 搭配使用。
