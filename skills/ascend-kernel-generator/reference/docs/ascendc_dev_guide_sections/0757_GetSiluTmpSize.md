###### GetSiluTmpSize

## 功能说明

用于获取 Silu Tiling 参数：Silu 接口能完成计算所需最大临时空间大小和最小临时空间大小。

## 函数原型

```cpp
inline void GetSiluTmpSize(const ge::Shape& srcShape, 
                          const uint32_t typeSize, 
                          const bool isReuseSource,
                          uint32_t& max, 
                          uint32_t& min)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| srcShape | 输入 | 输入的 shape 信息 |
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为 half，此处应传入 2，即 `sizeof(half)` |
| isReuseSource | 输入 | 是否复用源操作数输入的空间，与 kernel 侧接口一致 |
| max | 输出 | Silu 接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel 侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。最大空间大小为 0 表示计算不需要临时空间 |
| min | 输出 | Silu 接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。最小空间大小为 0 表示计算不需要临时空间 |

**说明：**
- max 仅作为参考值，有可能大于 Unified Buffer 剩余空间的大小，该场景下，开发者需要根据 Unified Buffer 剩余空间的大小来选取合适的临时空间大小

## 返回值说明

无

## 约束说明

无

## 调用示例

```cpp
// 输入shape信息为1024;算子输入的数据类型为half;不允许修改源操作数
std::vector<int64_t> shape_vec = {1024};
ge::Shape shape(shape_vec);
uint32_t max;
uint32_t min;
AscendC::GetSiluTmpSize(shape, 2, false, max, min);
```
