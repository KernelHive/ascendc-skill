## aclCreateTensorList

## 功能描述

创建 `aclTensorList` 对象，作为单算子 API 执行接口的入参。

`aclTensorList` 是框架定义的一种用来管理和存储多个张量数据的列表结构，开发者无需关注其内部实现，直接使用即可。

## 函数原型

```c
aclTensorList *aclCreateTensorList(const aclTensor *const *value, uint64_t size)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| value  | 输入      | `aclTensor` 类型的指针数组，其指向的值会赋给 `TensorList` |
| size   | 输入      | 张量列表的长度，取值为正整数 |

## 返回值说明

成功则返回创建好的 `aclTensorList`，否则返回 `nullptr`。

## 约束与限制

- 调用本接口前，需提前调用 `aclCreateTensor` 接口创建 `aclTensor`
- 本接口需与 `aclDestroyTensorList` 接口配套使用，分别完成 `aclTensorList` 的创建与销毁
- 调用 `aclGetTensorListSize` 接口可以获取 `aclTensorList` 的大小
- 调用如下接口可刷新输入/输出 `aclTensorList` 中记录的 Device 内存地址：
  - `aclSetDynamicInputTensorAddr`
  - `aclSetDynamicOutputTensorAddr`
  - `aclSetDynamicTensorAddr`

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```c
// 创建 aclTensor: input1 input2
std::vector<int64_t> shape = {1, 2, 3};
aclTensor *input1 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
                                    nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), 
                                    shape.size(), nullptr);
aclTensor *input2 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
                                    nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), 
                                    shape.size(), nullptr);

// 创建 aclTensorList
std::vector<aclTensor *> tmp{input1, input2};
aclTensorList* tensorList = aclCreateTensorList(tmp.data(), tmp.size());

// aclTensorList 作为单算子 API 执行接口的入参
auto ret = aclxxXxxGetWorkspaceSize(srcTensor, tensorList, ..., outTensor, ..., 
                                    &workspaceSize, &executor);
ret = aclxxXxx(...);
...

// 销毁 aclTensorList
ret = aclDestroyTensorList(tensorList);
```
