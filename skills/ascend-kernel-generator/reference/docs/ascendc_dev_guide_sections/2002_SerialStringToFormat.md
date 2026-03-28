##### SerialStringToFormat

## 函数功能

将字符串转化为 Format 类型值。

从 GCC 5.1 版本开始，libstdc++ 为了更好的实现 C++11 规范，更改了 `std::string` 和 `std::list` 的一些接口，导致新老版本 ABI 不兼容。所以推荐使用 `15.2.3.37.4 AscendStringToFormat` 替代本接口。

使用该接口需要包含 `type_utils.h` 头文件。

```cpp
#include "graph/utils/type_utils.h"
```

## 函数原型

```cpp
static Format SerialStringToFormat(const std::string &str)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| str  | 输入      | 待转换的 Format 字符串形式 |

## 返回值说明

输入合法时，返回转换后的 Format enum 值，枚举定义请参考 `15.2.3.59 Format`；输入不合法时，返回 `FORMAT_RESERVED`，并打印报错信息。

## 约束说明

无。

## 调用示例

```cpp
std::string format_str = "NHWC";
auto format = ge::TypeUtils::SerialStringToFormat(format_str); // 1
```
