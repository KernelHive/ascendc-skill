## aclCreateScalar

## 函数功能

创建 `aclScalar` 对象，作为单算子 API 执行接口的入参。

`aclScalar` 是框架定义的一种用来管理和存储标量数据的结构，开发者无需关注其内部实现，直接使用即可。

## 函数原型

```c
aclScalar *aclCreateScalar(void *value, aclDataType dataType)
```

## 参数说明

`aclDataType` 是框架定义的一种数据类型枚举类，具体参见《应用开发指南 (C&C++)》中“acl API参考 > 数据类型及其操作接口 > aclDataType”。

| 参数名   | 输入/输出 | 说明                                                         |
|----------|-----------|--------------------------------------------------------------|
| value    | 输入      | Host 侧的 scalar 类型的指针，其指向的值会作为 scalar         |
| dataType | 输入      | scalar 的数据类型                                            |

## 返回值说明

成功则返回创建好的 `aclScalar`，否则返回 `nullptr`。

## 约束与限制

- 本接口需与 `aclDestroyScalar` 接口配套使用，分别完成 `aclScalar` 的创建与销毁。
- 如需创建多个 `aclScalar` 对象，可调用 `aclCreateScalarList` 接口来存储标量列表。

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```c
// 创建 aclScalar
float alphaValue = 1.2f;
aclScalar* alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
...
// aclScalar 作为单算子 API 执行接口的入参
auto ret = aclxxXxxGetWorkspaceSize(srcTensor, alpha, ..., outTensor, ..., &workspaceSize, &executor);
ret = aclxxXxx(...);
...
// 销毁 aclScalar
ret = aclDestroyScalar(alpha);
```
