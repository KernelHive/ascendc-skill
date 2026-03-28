##### DataTypeToSerialString

## 函数功能

将 DataType 类型值转化为字符串表达。

从 GCC 5.1 版本开始，libstdc++ 为了更好的实现 C++11 规范，更改了 `std::string` 和 `std::list` 的一些接口，导致新老版本 ABI 不兼容。所以推荐使用 15.2.3.37.1 `DataTypeToAscendString` 替代本接口。

使用该接口需要包含 `type_utils.h` 头文件。

```cpp
#include "graph/utils/type_utils.h"
```

## 函数原型

```cpp
static std::string DataTypeToSerialString(const DataType data_type)
```

## 参数说明

| 参数      | 输入/输出 | 说明                                                         |
|-----------|-----------|--------------------------------------------------------------|
| data_type | 输入      | 待转换的 DataType，支持的 DataType 请参考 15.2.3.58 DataType |

## 返回值说明

转换后的 DataType 字符串。

## 约束说明

无。

## 调用示例

```cpp
DataType data_type = ge::DT_UINT32;
auto type_str = ge::TypeUtils::DataTypeToSerialString(data_type); // "DT_UINT32"
```
