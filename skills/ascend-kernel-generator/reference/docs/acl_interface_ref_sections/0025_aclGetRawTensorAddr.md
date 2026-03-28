## aclGetRawTensorAddr

## 功能描述
获取 `aclTensor` 中原始记录的 Device 内存地址。`aclTensor` 由 `aclCreateTensor` 接口创建。

## 函数原型
```c
aclnnStatus aclGetRawTensorAddr(const aclTensor *tensor, void **addr)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| tensor | 输入 | 输入的 `aclTensor` 指针 |
| addr   | 输入 | 返回的 `aclTensor` 中记录的 Device 内存地址 |

## 返回值
- 返回 `0` 表示成功
- 返回其他值表示失败，返回码列表参见公共接口返回码

### 可能失败的原因
- 返回 `161001`：参数 `tensor` 或 `addr` 为空指针

## 约束与限制
- 必须在一阶段接口 `aclxxXxxGetWorkspaceSize` 之前或二阶段接口 `aclxxXxx` 之后使用，不支持在一阶段与二阶段接口之间使用
- 本接口可与 `aclSetRawTensorAddr` 接口配套使用，查看刷新后的结果是否符合预期

## 调用示例
关键代码示例如下，仅供参考，不支持直接拷贝运行。

```c
// 创建输入和输出张量 inputTensor 和 outputTensor
std::vector<int64_t> shape = {1, 2, 3};
void *addr1;
void *addr2;
... // 申请 device 内存 addr1，addr2
aclTensor inputTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), addr1);
aclTensor outputTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), addr2);

void *getAddr1 = nullptr;
void *getAddr2 = nullptr;
// 获取 inputTensor 中记录的 device 内存地址，此处获取到的指针 getAddr1 指向的内存地址与 addr1 一致
auto ret = aclGetRawTensorAddr(inputTensor, &getAddr1);
// 获取 outputTensor 中记录的 device 内存地址，此处获取到的指针 getAddr2 指向的内存地址与 addr2 一致
auto ret = aclGetRawTensorAddr(outputTensor, &getAddr2);

// 调用 Xxx 算子一、二阶段接口
ret = aclxxXxxGetWorkspaceSize(inputTensor, outputTensor, &workspaceSize, &executor);
ret = aclxxXxx(workspace, workspaceSize, executor, stream);
...
```
