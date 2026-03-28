###### GetGeluMaxMinTmpSize

## 功能说明

Kernel 侧 Gelu、FasterGelu、FasterGeluV2 接口的计算需要开发者预留/申请临时空间。本接口用于在 Host 侧获取预留/申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为 Tiling 参数传递到 Kernel 侧使用。

- 为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小。
- 在最小临时空间-最大临时空间范围内，随着临时空间增大，Kernel 侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。

## 函数原型

### 获取最大临时空间大小
```cpp
uint32_t GetGeluMaxTmpSize(const ge::Shape& srcShape, const uint32_t typeSize)
```

### 获取最小临时空间大小
```cpp
uint32_t GetGeluMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize)
```

### 获取最大/最小临时空间
```cpp
inline void GetGeluMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, uint32_t& maxValue, uint32_t& minValue)
```

## 参数说明

| 参数 | 输入/输出 | 功能说明 |
|------|-----------|----------|
| srcShape | 输入 | 输入的 shape 信息 |
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为 half，此处应传入 2 |
| maxValue | 输出 | Gelu、FasterGelu、FasterGeluV2 接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，Kernel 侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。最大空间大小为 0 表示计算不需要临时空间。<br>**说明：** maxValue 仅作为参考值，有可能大于 Unified Buffer 剩余空间的大小，该场景下，开发者需要根据 Unified Buffer 剩余空间的大小来选取合适的临时空间大小 |
| minValue | 输出 | Gelu、FasterGelu、FasterGeluV2 接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。最小空间大小为 0 表示计算不需要临时空间 |

## 返回值说明

- `GetGeluMaxTmpSize`：返回 Kernel 侧接口能完成计算所需最大临时空间大小
- `GetGeluMinTmpSize`：返回 Kernel 侧接口能完成计算所需最小临时空间大小  
- `GetGeluMaxMinTmpSize`：无返回值

## 约束说明

无

## 调用示例

```cpp
// 输入 shape 信息为 1024; 算子输入的数据类型为 half;
std::vector<int64_t> shape_vec = {1024};
ge::Shape srcShape(shape_vec);
uint32_t typeSize = 2;
uint32_t maxValue = 0;
uint32_t minValue = 0;
AscendC::GetGeluMaxMinTmpSize(srcShape, typeSize, maxValue, minValue);
```
