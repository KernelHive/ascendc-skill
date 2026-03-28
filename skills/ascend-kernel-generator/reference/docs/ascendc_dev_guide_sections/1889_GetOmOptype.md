##### GetOmOptype

## 函数功能
获取模型的算子类型。

## 函数原型
```cpp
std::string GetOmOptype() const
Status GetOmOptype(ge::AscendString& om_op_type) const
```

## 须知
数据类型为 string 的接口后续版本会废弃，建议使用数据类型为非 string 的接口。

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| om_op_type | 输出 | 模型的算子类型。 |

## 约束说明
无。
