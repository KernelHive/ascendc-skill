###### GetTanhTmpBufferFactorSize

## 功能说明

该接口用于获取 `maxLiveNodeCount` 和 `extraBuf`，在固定空间大小的情况下，通过这两个参数可以推算算子单次最大计算元素数量。

- `maxLiveNodeCount`：表示临时空间是单次计算数据量所占空间的多少倍
- `extraBuf`：表示使用的额外临时空间大小

## 推算示例

### 调用单个接口

算子实现需要调用 Tanh 接口，开发者为其预留 `currBuff` 大小的空间，利用 `GetTanhTmpBufferFactorSize` 接口得到 `maxLiveNodeCount`、`extraBuf` 输出值，可推导算子单次最大计算元素数量为：

```cpp
currentShapeSize = (currBuff - extraBuf) / maxLiveNodeCount / typeSize
```

### 调用多个接口

算子实现需要调用两个 kernel 侧 API `KernelIntf1`、`KernelIntf2`，利用两个 `GetXxxTmpBufferFactorSize`（其中 Xxx 为需要调用的两个高阶 API）接口的两组输出值 `(maxLiveNodeCount, extraBuf)` 以及当前现有的临时空间，推导单次最大计算元素数量 `currentShapeSize` 为：

```cpp
currentShapeSize1 = (currBuff - extraBuf1) / maxLiveNodeCount1 / typeSize
currentShapeSize2 = (currBuff - extraBuf2) / maxLiveNodeCount2 / typeSize
currentShapeSize = min(currentShapeSize1, currentShapeSize2)
```

**注意：**
- `currBuff` 表示接口计算可用的空间，需要去除用户输入输出等空间
- 接口获取的 `maxLiveNodeCount` 值可能为 0，计算时需要判断该值非 0，避免除零错误

## 函数原型

```cpp
void GetTanhTmpBufferFactorSize(const uint32_t typeSize, uint32_t &maxLiveNodeCount, uint32_t &extraBuf)
```

## 参数说明

| 参数名 | 输入/输出 | 功能说明 |
|--------|-----------|----------|
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为 half，此处应传入 2 |
| maxLiveNodeCount | 输出 | 最大存活节点数，表示临时空间是单次计算数据量所占空间的多少倍 |
| extraBuf | 输出 | 使用的额外临时空间大小，单位为字节 |

## 返回值说明

无

## 约束说明

当利用 `maxLiveNodeCount`、`extraBuf` 反推出的 `currentShapeSize * typeSize < 256B` 时，`currentShapeSize` 按照 `256B/typeSize` 的值向上取整。

## 调用示例

完整的调用样例请参考 15.1.5.1.31 更多样例。

```cpp
uint32_t maxLiveNodeCount = 0;
uint32_t extraBuf = 0;
AscendC::GetTanhTmpBufferFactorSize(typeSize, maxLiveNodeCount, extraBuf);
```
