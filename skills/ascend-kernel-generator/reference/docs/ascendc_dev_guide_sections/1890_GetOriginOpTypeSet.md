##### GetOriginOpTypeSet

## 函数功能

获取原始模型的算子类型集合。

## 函数原型

```cpp
std::set<std::string> GetOriginOpTypeSet() const
Status GetOriginOpTypeSet(std::set<ge::AscendString>& ori_op_type) const
```

## 须知

数据类型为 string 的接口后续版本会废弃，建议使用数据类型为非 string 的接口。

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| ori_op_type | 输出 | 原始模型的算子类型集合 |

## 约束说明

无。
