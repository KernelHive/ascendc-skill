## aclCreateScalarList

## 函数功能

创建 `aclScalarList` 对象，作为单算子 API 执行接口的入参。

`aclScalarList` 是框架定义的一种用来管理和存储多个标量数据的列表结构，开发者无需关注其内部实现，直接使用即可。

## 函数原型

```c
aclScalarList *aclCreateScalarList(const aclScalar *const *value, uint64_t size)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| value  | 输入      | 指向 `aclScalar` 指针数组首地址的指针，数组内 `aclScalar` 指针会依次拷贝给 `aclScalarList`。 |
| size   | 输入      | 标量列表的长度，取值为正整数。 |

## 返回值说明

成功则返回创建好的 `aclScalarList`，否则返回 `nullptr`。

## 约束与限制

- 调用本接口前，需提前调用 `aclCreateScalar` 接口创建 `aclScalar`。
- 本接口需与 `aclDestroyScalarList` 接口配套使用，分别完成 `aclScalarList` 的创建与销毁。
- 调用 `aclGetScalarListSize` 接口可以获取 `aclScalarList` 的大小。

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```c
// 创建alpha1 aclScalar
float alpha1Value = 1.2f;
aclScalar *alpha1 = aclCreateScalar(&alpha1Value, aclDataType::ACL_FLOAT);

// 创建alpha2 aclScalar
float alpha2Value = 2.2f;
aclScalar *alpha2 = aclCreateScalar(&alpha2Value, aclDataType::ACL_FLOAT);

// 创建aclScalarList
std::vector<aclScalar *> tempscalar{alpha1, alpha2};
aclScalarList *scalarlist = aclCreateScalarList(tempscalar.data(), tempscalar.size());

...
// aclScalarList作为单算子API执行接口的入参
auto ret = aclxxXxxGetWorkspaceSize(srcTensor, scalarlist, ..., outTensor, ..., &workspaceSize, &executor);
ret = aclxxXxx(...);
...

// 销毁aclScalarList
ret = aclDestroyScalarList(scalarlist);
```
