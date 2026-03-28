##### TryGetInputDesc

## 函数功能

根据算子 Input 名称获取算子 Input 的 TensorDesc。

## 函数原型

```cpp
graphStatus TryGetInputDesc(const std::string &name, TensorDesc &tensor_desc) const
graphStatus TryGetInputDesc(const char_t *name, TensorDesc &tensor_desc) const
```

## 须知

数据类型为 string 的接口后续版本会废弃，建议使用数据类型为非 string 的接口。

## 参数说明

| 参数名      | 输入/输出 | 描述                                                         |
|-------------|-----------|-------------------------------------------------------------|
| name        | 输入      | 算子的 Input 名。                                           |
| tensor_desc | 输出      | 返回算子端口的当前设置格式，为 TensorDesc 对象。            |

## 返回值

graphStatus 类型：
- True：有此端口，获取 TensorDesc 成功
- False：无此端口，出参为空，获取 TensorDesc 失败

## 异常处理

| 异常场景       | 说明         |
|----------------|--------------|
| 无对应 name 输入 | 返回 False。 |

## 约束说明

无。
