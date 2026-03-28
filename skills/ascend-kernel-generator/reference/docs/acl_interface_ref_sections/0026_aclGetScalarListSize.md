## aclGetScalarListSize

## 函数功能
获取 `aclScalarList` 的大小。`aclScalarList` 通过 `aclCreateScalarList` 接口创建。

## 函数原型
```c
aclnnStatus aclGetScalarListSize(const aclScalarList *scalarList, uint64_t *size)
```

## 参数说明

| 参数名     | 输入/输出 | 说明                     |
|------------|-----------|--------------------------|
| scalarList | 输入      | 输入的 `aclScalarList`   |
| size       | 输出      | 返回 `aclScalarList` 的大小 |

## 返回值说明
- 返回 `0` 表示成功，返回其他值表示失败。返回码列表参见“公共接口返回码”。
- 可能失败的原因：
  - 返回 `161001`：参数 `scalarList` 或 `size` 为空指针。

## 约束与限制
无

## 调用示例
关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 创建 aclScalarList
std::vector<aclScalar *> tempscalar{alpha1, alpha2};
aclScalarList *scalarlist = aclCreateScalarList(tempscalar.data(), tempscalar.size());

// 获取 scalarList 的大小
uint64_t size = 0;
auto ret = aclGetScalarListSize(scalarList, &size); // 这里获取到的 scalarList 的 size 为 2

// 销毁 aclScalarList
ret = aclDestroyScalarList(scalarlist);
```
