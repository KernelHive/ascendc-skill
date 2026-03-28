###### GetMeanTmpBufferFactorSize

## 功能说明

该接口用于获取 `maxLiveNodeCnt` 和 `extraBuf`，在固定空间大小的情况下，通过这两个参数可以推算算子单次最大计算元素数量。

- `maxLiveNodeCnt` 表示临时空间是单次计算数据量所占空间的多少倍
- `extraBuf` 表示使用的额外临时空间大小

## 推算示例

### 调用单个高阶 API

算子实现需要调用 `MeanMax`/`ClampMin` 接口，开发者为其预留 `currBuff` 大小的空间，利用 `GetMeanTmpBufferFactorSize` 接口得到 `maxLiveNodeCnt`、`extraBuf` 输出值，可推导算子单次最大计算元素数量为：

```cpp
currentShapeSize = (currBuff - extraBuf) / maxLiveNodeCnt / typeSize
```

### 调用多个高阶 API

算子实现需要调用两个 kernel 侧 API `KernelIntf1`、`KernelIntf2`，利用两个 `GetXxxTmpBufferFactorSize`（其中 `Xxx` 为需要调用的两个高阶 API）接口的两组输出值 `(maxLiveNodeCnt, extraBuf)` 以及当前现有的临时空间，推导单次最大计算元素数量 `currentShapeSize` 为：

```cpp
currentShapeSize1 = (currBuff - extraBuf1) / maxLiveNodeCnt1 / typeSize
currentShapeSize2 = (currBuff - extraBuf2) / maxLiveNodeCnt2 / typeSize
currentShapeSize = min(currentShapeSize1, currentShapeSize2)
```

> **注意**：上文中的 `currBuff` 表示接口计算可用的空间，需要去除用户输入输出等空间。

## 函数原型

```cpp
void GetMeanTmpBufferFactorSize(const uint32_t typeSize, uint32_t &maxLiveNodeCount, uint32_t &extraBuffer)
```

## 参数说明

| 参数 | 输入/输出 | 功能描述 |
|------|-----------|----------|
| `typeSize` | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为 `half`，此处应传入 `2` |
| `maxLiveNodeCount` | 输出 | 最大存活节点数，表示临时空间是单次计算数据量所占空间的多少倍 |
| `extraBuffer` | 输出 | 使用的额外临时空间大小，单位为 byte |

## 返回值说明

无

## 约束说明

当利用 `maxLiveNodeCount`、`extraBuffer` 反推出的 `currentShapeSize * typeSize < 256B` 时，`currentShapeSize` 按照 `256B/typeSize` 的值向上取整。

## 调用示例

完整的调用样例请参考 15.1.5.1.31 更多样例。

```cpp
uint32_t maxLiveNodeCnt = 0;
uint32_t extraBuf = 0;
AscendC::GetMeanTmpBufferFactorSize(typeSize, maxLiveNodeCnt, extraBuf);
```
