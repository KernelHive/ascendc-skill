## aclGetBoolArraySize

## 函数功能

获取 `aclBoolArray` 的大小。`aclBoolArray` 通过 `aclCreateBoolArray` 接口创建。

## 函数原型

```c
aclnnStatus aclGetBoolArraySize(const aclBoolArray *array, uint64_t *size)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| array  | 输入      | 输入的 `aclBoolArray` |
| size   | 输出      | 返回 `aclBoolArray` 的大小 |

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
// 创建 aclBoolArray
std::vector<bool> maskData = {true, false};
aclBoolArray *mask = aclCreateBoolArray(maskData.data(), maskData.size());

// 使用 aclGetBoolArraySize 接口获取 mask 的大小
uint64_t size = 0;
auto ret = aclGetBoolArraySize(mask, &size); // 获取到的 mask 的 size 为 2

// 销毁 aclBoolArray
ret = aclDestroyBoolArray(mask);
```
