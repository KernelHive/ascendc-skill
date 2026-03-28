## aclCreateIntArray

## 函数功能

创建 `aclIntArray` 对象，作为单算子 API 执行接口的入参。

`aclIntArray` 是框架定义的一种用来管理和存储整型数据的数组结构，开发者无需关注其内部实现，直接使用即可。

## 函数原型

```c
aclIntArray *aclCreateIntArray(const int64_t *value, uint64_t size)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| value  | 输入      | Host 侧的 `int64_t` 类型的指针，其指向的值会拷贝给 `aclIntArray`。 |
| size   | 输入      | 整型数组的长度，取值为正整数。 |

## 返回值说明

成功则返回创建好的 `aclIntArray`，否则返回 `nullptr`。

## 约束与限制

- 本接口需与 `aclDestroyIntArray` 接口配套使用，分别完成 `aclIntArray` 的创建与销毁。
- 调用 `aclGetIntArraySize` 接口可以获取 `aclIntArray` 的大小。

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```c
// 创建 aclIntArray
std::vector<int64_t> sizeData = {1, 1, 2, 3};
aclIntArray *size = aclCreateIntArray(sizeData.data(), sizeData.size());
...
// aclIntArray 作为单算子 API 执行接口的入参
auto ret = aclxxXxxGetWorkspaceSize(srcTensor, size, ..., outTensor, ..., &workspaceSize, &executor);
ret = aclxxXxx(...);
...
// 销毁 aclIntArray
ret = aclDestroyIntArray(size);
```
