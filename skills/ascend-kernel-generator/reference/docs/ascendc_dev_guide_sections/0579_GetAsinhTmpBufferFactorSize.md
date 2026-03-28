###### GetAsinhTmpBufferFactorSize

## 功能说明

kernel侧Asinh接口的计算需要开发者预留/申请临时空间，最大临时空间（maxTmpBuffer）和输入所占空间（inputSize * typeSize）存在以下关系：

```
maxTmpBuffer = maxLiveNodeCount * inputSize * typeSize + extraBuffer
```

其中：
- `maxLiveNodeCount` 表示最大临时空间是输入所占空间的多少倍
- `extraBuffer` 表示使用的额外临时空间大小

该接口用于获取 `maxLiveNodeCount` 和 `extraBuffer`，在固定空间大小的情况下，通过 `maxLiveNodeCount` 和 `extraBuffer` 可以推算算子单次最大计算元素数量。

**注意**：接口获取的 `maxLiveNodeCount` 值可能为0，计算时需要判断该值非0，避免除零错误。

示例如下：

算子实现需要调用Asinh接口，开发者为其预留 `currBuff` 大小的空间，利用 `GetAsinhTmpBufferFactorSize` 接口得到 `maxLiveNodeCount`、`extraBuffer` 输出值，反推Asinh算子单次最大计算元素数量为：

```
currentShapeSize = (currBuff - extraBuffer) / maxLiveNodeCount / typeSize
```

## 函数原型

```cpp
void GetAsinhTmpBufferFactorSize(const uint32_t typeSize, uint32_t &maxLiveNodeCount, uint32_t &extraBuf)
```

## 参数说明

**表 参数列表**

| 参数名 | 输入/输出 | 功能描述 |
|--------|-----------|----------|
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。 |
| maxLiveNodeCount | 输出 | 最大存活节点数，最大临时空间是输入所占空间的多少倍。 |
| extraBuf | 输出 | 使用的额外临时空间大小，单位为字节。 |

## 返回值说明

无

## 约束说明

当利用 `maxLiveNodeCount`、`extraBuf` 反推出的 `currentShapeSize * typeSize < 256B` 时，`currentShapeSize` 按照 `256B/typeSize` 的值向上取整。

## 调用示例

```cpp
// 获取输入类型为half的Asinh操作的maxLiveNodeCount和extraBuffer
uint32_t typeSize = sizeof(half);
uint32_t maxLiveNodeCount = 0;
uint32_t extraBuffer = 0;

AscendC::GetAsinhTmpBufferFactorSize(typeSize, maxLiveNodeCount, extraBuffer);
```
