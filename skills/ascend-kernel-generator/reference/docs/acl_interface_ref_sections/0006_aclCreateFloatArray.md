## aclCreateFloatArray

## 函数功能

创建 `aclFloatArray` 对象，作为单算子 API 执行接口的入参。

`aclFloatArray` 是框架定义的一种用来管理和存储浮点型数据的数组结构，开发者无需关注其内部实现，直接使用即可。

## 函数原型

```c
aclFloatArray *aclCreateFloatArray(const float *value, uint64_t size)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|------------|------|
| value  | 输入       | Host 侧的 float 类型指针，其指向的值会拷贝给 `aclFloatArray`。 |
| size   | 输入       | 浮点型数组的长度，取值为正整数。 |

## 返回值说明

成功则返回创建好的 `aclFloatArray`，否则返回 `nullptr`。

## 约束与限制

- 本接口需与 `aclDestroyFloatArray` 接口配套使用，分别完成 `aclFloatArray` 的创建与销毁。
- 调用 `aclGetFloatArraySize` 接口可以获取 `aclFloatArray` 的大小。

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```c
// 创建 aclFloatArray
std::vector<float> scalesData = {1.0, 1.0, 2.0, 2.0};
aclFloatArray *scales = aclCreateFloatArray(scalesData.data(), scalesData.size());
...

// aclFloatArray 作为单算子 API 执行接口的入参
auto ret = aclxxXxxGetWorkspaceSize(srcTensor, scales, ..., outTensor, ..., &workspaceSize, &executor);
ret = aclxxXxx(...);
...
// 销毁 aclFloatArray
ret = aclDestroyFloatArray(scales);
```
