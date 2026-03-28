##### GetDynamicOutputNum

## 函数功能

获取算子的动态 Output 的实际个数。

## 函数原型

```cpp
int32_t GetDynamicOutputNum(const std::string &name) const
int32_t GetDynamicOutputNum(const char_t *name) const
```

## 须知

数据类型为 string 的接口后续版本会废弃，建议使用数据类型为非 string 的接口。

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| name   | 输入      | 算子的动态 Output 名。 |

## 返回值

实际动态 Output 的个数。  
当 name 非法，或者算子无动态 Output 时，返回 0。

## 约束说明

无。
