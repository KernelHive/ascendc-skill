## aclGetIntArraySize

## 函数功能
获取 `aclIntArray` 的大小。`aclIntArray` 通过 `aclCreateIntArray` 接口创建。

## 函数原型
```c
aclnnStatus aclGetIntArraySize(const aclIntArray *array, uint64_t *size)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| array  | 输入      | 输入的 `aclIntArray` |
| size   | 输出      | 返回 `aclIntArray` 的大小 |

## 返回值说明
- 返回 `0` 表示成功
- 返回其他值表示失败，返回码列表参见“公共接口返回码”

### 可能失败的原因
- 返回 `161001`：参数 `array` 或 `size` 为空指针

## 约束与限制
无

## 调用示例
关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 创建 aclIntArray
std::vector<int64_t> valueData = {1, 1, 2, 3};
aclIntArray *valueArray = aclCreateIntArray(sizeData.data(), sizeData.size());

// 使用 aclGetIntArraySize 接口获取 valueArray 的大小
uint64_t size = 0;
auto ret = aclGetIntArraySize(valueArray, &size); // 获取到的 valueArray 的 size 为 4

// 销毁 aclIntArray
ret = aclDestroyIntArray(valueArray);
```
