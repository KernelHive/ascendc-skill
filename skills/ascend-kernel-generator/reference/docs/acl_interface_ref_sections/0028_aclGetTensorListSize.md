## aclGetTensorListSize

## 函数功能
获取 `aclTensorList` 的大小，`aclTensorList` 通过 `aclCreateTensorList` 接口创建。

## 函数原型
```c
aclnnStatus aclGetTensorListSize(const aclTensorList *tensorList, uint64_t *size)
```

## 参数说明

| 参数名     | 输入/输出 | 说明                     |
|------------|-----------|--------------------------|
| tensorList | 输入      | 输入的 `aclTensorList`   |
| size       | 输出      | 返回 `aclTensorList` 的大小 |

## 返回值说明
- 返回 `0` 表示成功，返回其他值表示失败，返回码列表参见“公共接口返回码”。
- 可能失败的原因：
  - 返回 `161001`：参数 `tensorList` 或 `size` 为空指针。

## 约束与限制
无

## 调用示例
关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 创建 aclTensorList
std::vector<aclTensor *> tmp{input1, input2};
aclTensorList* tensorList = aclCreateTensorList(tmp.data(), tmp.size());

// 获取 tensorList 的大小
uint64_t size = 0;
auto ret = aclGetTensorListSize(tensorList, &size); // 这里获取到的 tensorList 的 size 为 2

// 销毁 aclTensorList
ret = aclDestroyTensorList(tensorList);
```
