## aclCreateBoolArray

本项目是CANN提供的图像处理、目标检测相关算子库，实现网络在NPU上加速计算。

本项目是CANN提供的transformer类大模型算子库，实现网络在NPU上加速计算。

## 函数功能

创建aclBoolArray对象，作为单算子API执行接口的入参。

aclBoolArray是框架定义的一种用来管理和存储布尔型数据的数组结构，开发者无需关注其内部实现，直接使用即可。

## 函数原型

```c
aclBoolArray *aclCreateBoolArray(const bool *value, uint64_t size)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| value  | 输入      | Host侧的bool类型指针，其指向的值会拷贝给aclBoolArray。 |
| size   | 输入      | 布尔型数组的长度，取值为正整数。 |

## 返回值说明

成功则返回创建好的aclBoolArray，否则返回nullptr。

## 约束与限制

- 本接口需与 `aclDestroyBoolArray` 接口配套使用，分别完成aclBoolArray的创建与销毁。
- 调用 `aclGetBoolArraySize` 接口可以获取aclBoolArray的大小。

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```c
// 创建aclBoolArray
std::vector<bool> maskData = {true, false};
aclBoolArray *mask = aclCreateBoolArray(maskData.data(), maskData.size());
...
// aclBoolArray作为单算子API执行接口的入参
auto ret = aclxxXxxGetWorkspaceSize(srcTensor, mask, ..., outTensor, ..., &workspaceSize, &executor);
ret = aclxxXxx(...);
...
// 销毁aclBoolArray
ret = aclDestroyBoolArray(mask);
```
