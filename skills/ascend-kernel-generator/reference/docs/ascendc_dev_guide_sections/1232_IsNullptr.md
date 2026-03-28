###### IsNullptr

## 支持的产品型号

- Atlas 训练系列产品
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品
- Atlas 推理系列产品

## 功能说明

判断输入的指针是否为空。若为空指针返回 true、并打印错误日志，否则返回 false。

## 函数原型

```cpp
static inline bool IsNullptr(const aclTensor *tensor, const char *name)
static inline bool IsNullptr(const aclTensorList *tensorList, const char *name)
static inline bool IsNullptr(const aclScalar *scalar, const char *name)
static inline bool IsNullptr(const aclIntArray *intArr, const char *name)
static inline bool IsNullptr(const aclBoolArray *boolArr, const char *name)
static inline bool IsNullptr(const aclFloatArray *floatArr, const char *name)
```

## 参数说明

| 参数   | 输入/输出 | 说明                                                                                 |
|--------|-----------|--------------------------------------------------------------------------------------|
| tensor | 输入      | 需要被检查的指针，支持 `aclTensor *`、`aclTensorList *`、`aclScalar *`、`aclIntArray *`、`aclBoolArray *`、`aclFloatArray *` 类型。 |
| name   | 输入      | 被检查的指针的一个标识，如果被检查指针为空，则打印的错误日志里会输出此标识（name）。 |

## 返回值说明

返回 bool 类型，如果指针被判断为 nullptr，返回 true，否则返回 false。

## 约束说明

无

## 调用示例

```cpp
#define OP_CHECK_NULL(param, retExpr) \
    if (IsNullptr(param, #param)) { \
        retExpr; \
    }
```
