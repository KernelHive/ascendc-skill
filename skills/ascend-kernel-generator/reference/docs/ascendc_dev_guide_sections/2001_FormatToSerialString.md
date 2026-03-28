##### FormatToSerialString

## 功能描述

将 Format 类型值转换为字符串表示。

> **注意**：从 GCC 5.1 版本开始，libstdc++ 为了更好实现 C++11 规范，更改了 `std::string` 和 `std::list` 的一些接口，导致新老版本 ABI 不兼容。因此推荐使用 `FormatToAscendString` 替代本接口。

使用该接口需要包含头文件：

```cpp
#include "graph/utils/type_utils.h"
```

## 函数原型

```cpp
static std::string FormatToSerialString(const Format format)
```

## 参数说明

| 参数   | 输入/输出 | 说明                                                         |
|--------|-----------|--------------------------------------------------------------|
| format | 输入      | 待转换的 Format，支持的 Format 请参考 15.2.3.59 Format 章节。 |

## 返回值

转换后的 Format 字符串。

## 约束说明

无。

## 调用示例

```cpp
ge::Format format = ge::Format::FORMAT_NHWC;
auto format_str = ge::TypeUtils::FormatToSerialString(format);  // "NHWC"
```
