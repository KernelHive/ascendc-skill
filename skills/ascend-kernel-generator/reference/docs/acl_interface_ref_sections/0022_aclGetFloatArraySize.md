## aclGetFloatArraySize

## 函数功能

获取 `aclFloatArray` 的大小。`aclFloatArray` 通过 `aclCreateFloatArray` 接口创建。

## 函数原型

```c
aclnnStatus aclGetFloatArraySize(const aclFloatArray *array, uint64_t *size)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| array  | 输入      | 输入的 `aclFloatArray` |
| size   | 输出      | 返回 `aclFloatArray` 的大小 |

## 返回值说明

返回 `0` 表示成功，返回其他值表示失败。返回码列表参见“公共接口返回码”。

可能失败的原因：

- 返回 `161001`：参数 `array` 或 `size` 为空指针。

## 约束与限制

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```c
// 创建 aclFloatArray
std::vector<float> scalesData = {1.0, 1.0, 2.0, 2.0};
aclFloatArray *scales = aclCreateFloatArray(scalesData.data(), scalesData.size());

// 使用 aclGetFloatArraySize 接口获取 scales 的大小
uint64_t size = 0;
auto ret = aclGetFloatArraySize(scales, &size); // 获取到的 scales 的 size 为 4

// 销毁 aclFloatArray
ret = aclDestroyFloatArray(scales);
```
