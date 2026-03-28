###### LogSoftMax Tiling

## 功能说明

Kernel侧LogSoftMax接口的计算需要开发者预留/申请临时空间。以下接口用于在Host侧获取预留/申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小，并调用`LogSoftMaxTilingFunc`函数获取`reduceSize`、`splitSize`等参数，作为Tiling参数传递到Kernel侧使用。

- 为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小
- 在最小临时空间-最大临时空间范围内，随着临时空间增大，Kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请

## 函数原型

### 获取临时空间大小接口

```cpp
uint32_t GetLogSoftMaxMaxTmpSize(const ge::Shape srcShape, const uint32_t dataTypeSize, const bool isReuseSource)
uint32_t GetLogSoftMaxMinTmpSize(const ge::Shape srcShape, const uint32_t dataTypeSize, const bool isReuseSource)
```

### Tiling计算接口

```cpp
void LogSoftMaxTilingFunc(const ge::Shape srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize, optiling::LogSoftMaxTiling& softmaxTiling)
```

## 参数说明

### GetLogSoftMaxMaxTmpSize/GetLogSoftMaxMinTmpSize 接口参数

| 参数名 | 输入/输出 | 功能说明 |
|--------|-----------|----------|
| srcShape | 输入 | 输入的shape信息 |
| dataTypeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2 |
| isReuseSource | 输入 | 是否复用源操作数输入的空间，与LogSoftMax接口一致 |

### LogSoftMaxTilingFunc 接口参数

| 参数名 | 输入/输出 | 功能说明 |
|--------|-----------|----------|
| srcShape | 输入 | 输入的shape信息 |
| dataTypeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2 |
| localWorkSpaceSize | 输入 | 输入的临时空间大小 |
| softmaxTiling | 输出 | 传递到kernel侧使用的Tiling参数 |

## 返回值说明

- `GetLogSoftMaxMaxTmpSize`/`GetLogSoftMaxMinTmpSize`接口返回值为最大/最小临时空间
- `LogSoftMaxTilingFunc`接口无返回值

## 约束说明

无

## 调用示例

```cpp
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    std::vector<int64_t> srcDims = {outter, inner};
    ge::Shape shape(srcDims);
    const uint32_t tmpsize = AscendC::GetLogSoftMaxMaxTmpSize(shape, dtypesize, false);
    AscendC::LogSoftMaxTilingFunc(shape, dtypesize, tmpsize, tiling.logSoftmaxTilingData);
    // ...
}
```
