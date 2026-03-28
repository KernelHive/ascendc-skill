##### CompiledInfo

## 函数功能

设置算子的 CompiledInfo 指针，用于存储 TilingParse 的结果即 CompileInfo 信息。

## 函数原型

```cpp
OpTilingParseContextBuilder &CompiledInfo(const void *compile_info)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| compile_info | 输入 | 编译信息指针 |

## 返回值说明

OpTilingParseContextBuilder 对象本身，用于链式调用。

## 约束说明

通过指针传入的参数（void*），其内存所有权归调用者所有；调用者必须确保指针在 ContextHolder 对象的生命周期内有效。
