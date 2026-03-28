##### SerialStringToDataType

## 功能描述

将 DataType 字符串表达转化为 DataType 类型值。

> **注意**：从 GCC 5.1 版本开始，libstdc++ 为了更好的实现 C++11 规范，更改了 `std::string` 和 `std::list` 的一些接口，导致新老版本 ABI 不兼容。所以推荐使用 `AscendStringToDataType` 替代本接口。

## 头文件

使用该接口需要包含以下头文件：

```cpp
#include "graph/utils/type_utils.h"
```

## 函数原型

```cpp
static DataType SerialStringToDataType(const std::string &str)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| str  | 输入      | 待转换的 DataType 字符串形式 |

## 返回值

- 输入合法时，返回转换后的 DataType 枚举值（枚举定义请参考 DataType）
- 输入不合法时，返回 `DT_UNDEFINED` 并打印报错日志

## 约束说明

无。

## 调用示例

```cpp
std::string type_str = "DT_UINT32";
auto data_type = ge::TypeUtils::SerialStringToDataType(type_str); // 8
```
