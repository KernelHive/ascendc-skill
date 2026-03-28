### 避免 Unified Buffer 的 bank 冲突

【优先级】高

## 说明

该性能优化指导适用于如下产品型号：

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 描述

为了提高数据访问的效率和吞吐量，Unified Buffer 采用了 bank（大小相等的内存模块）结构设计。Unified Buffer 总大小为 192K，划分为 48 个 bank。每个 bank 由 128 行组成，每行长度为 32B。这 48 个 bank 进一步组织为 16 个 bank group，每个 bank group 包含 3 个 bank，例如 bank15、bank31 和 bank47 组成一个 bank group。

图 5-14 bank 结构示意图（图中箭头方向表示内存排布的顺序）

每个 bank 可以独立地进行数据的读写操作，允许多个数据请求同时进行。然而，当多个读写操作试图同时访问同一个 bank 或 bank group 时，由于硬件资源的限制，这些操作必须排队等待，会导致 bank 冲突，引起性能下降。

具体来说，Vector 计算单元每拍（一个指令周期）能够从每个 bank group 中读取或写入一行数据。如果同一个 API 中的多个操作试图同时访问同一个 bank 或 bank group，Vector 计算单元无法在同一个周期内处理所有请求，导致这些请求排队等待。这种排队增加了数据访问的延迟，降低了系统的整体性能。

## bank 冲突的典型场景

bank 冲突主要可以分为以下三种场景：

- **读写冲突**：读操作和写操作同时尝试访问同一个 bank。
- **写写冲突**：多个写操作同时尝试访问同一个 bank group。
- **读读冲突**：多个读操作同时尝试访问同一个 bank group。

下文给出了一些具体的示例，假设，0x10000 地址在 bank16 上，0x10020 在 bank17 上，0x20020 在 bank33 上，如下图所示：

图 5-15 地址分配示意图

### 读写冲突示例

Vector 指令的源操作数 src 和目的操作数 dst 同时读写到同一个 bank 时造成读写冲突，具体分析如下：

表 5-2 读写冲突示例

| 序号 | src 地址 | dst 地址 | bank              | bank group        | 结论                                                         |
|------|----------|----------|-------------------|-------------------|--------------------------------------------------------------|
| 示例1 | 0x10020  | 0x10000  | bank_id0 != bank_id1 | bank_group_id0 != bank_group_id1 | src 地址和 dst 地址分别属于 bank16 和 bank17，故无冲突。     |
| 示例2 | 0x10020  | 0x10E20  | bank_id0 == bank_id1 | bank_group_id0 == bank_group_id1 | src 地址和 dst 地址的地址都在 bank17，故存在冲突。           |

### 写写冲突示例

Vector 指令目的操作数 dst 对应的 8 个 DataBlock（block0-block7）同时写到一个 bank group 时造成写写冲突，具体分析如下：

表 5-3 写写冲突示例

| 序号 | dst 地址 | blk_stride | block0_addr | block1_addr | block2_addr | ... | 结论                                                         |
|------|----------|------------|-------------|-------------|-------------|-----|--------------------------------------------------------------|
| 示例1 | 0x1FE00  | 16         | 0x1FE00     | 0x20000     | 0x20200     | ... | 8 个 DataBlock 均在一个 bank group 下，故全部冲突，8 拍完成一个 Repeat 的写入。 |
| 示例2 | 0x1FE00  | 8          | 0x1FE00     | 0x1FF00     | 0x20000     | ... | block0 和 block2 在一个 bank group，存在冲突，4 拍完成一个 Repeat 的写入。 |

### 读读冲突

#### Vector 指令多个源操作数同时读到同一个 bank group 时造成读读冲突

表 5-4 双 src 场景读读冲突示例

| 序号 | src0 地址 | src1 地址 | bank              | bank group        | 结论     |
|------|-----------|-----------|-------------------|-------------------|----------|
| 示例1 | 0x10020   | 0x20020   | bank_id0 != bank_id1 | bank_group_id0 == bank_group_id1 | 存在冲突 |
| 示例2 | 0x10020   | 0x10000   | bank_id0 != bank_id1 | bank_group_id0 != bank_group_id1 | 无冲突   |

#### Vector 指令某一个源操作数对应的 8 个 DataBlock（block0-block7）读到同一个 bank group 时造成读读冲突

表 5-5 单 src 场景读读冲突示例

| 序号 | src 地址 | blk_stride | block0_addr | block1_addr | block2_addr | ... | 结论                                                         |
|------|----------|------------|-------------|-------------|-------------|-----|--------------------------------------------------------------|
| 示例1 | 0x1FE00  | 16         | 0x1FE00     | 0x20000     | 0x20200     | ... | 8 个 DataBlock 均在一个 bank group 下，故全部冲突，8 拍完成一个 Repeat 的读操作。 |
| 示例2 | 0x1FE00  | 8          | 0x1FE00     | 0x1FF00     | 0x20000     | ... | block0 和 block2 在同一个 bank group 下，存在冲突，4 拍完成一个 Repeat。 |

## 说明

通过 msProf 工具可以进行资源冲突占比的相关性能数据采集。工具的具体使用方法和资源冲突占比文件性能数据文件说明请参考《算子开发工具用户指南》。

## 如何避免 bank 冲突

避免 bank 冲突的方法有两种：优化计算逻辑和优化地址分配。

### 优化计算逻辑

对一个 shape 为 (8, 16, 16) 的输入做 (1, 0, 2) 的 transpose 操作，输出 shape 为 (16, 8, 16)。通过将计算逻辑由“跳读，连续写”修改为“连续读，跳写”可避免冲突问题。实现方案对比如下：

| 实现方案 | 原始实现 | 优化实现 |
|----------|----------|----------|
| 实现方法 | 跳读，连续写<br>同一 Repeat 内输入的 8 个 DataBlock 都在同一个 bank group 而发生读读冲突。 | 连续读，跳写<br>同一个 Repeat 内输入的 8 个 DataBlock 不在同一个 bank group 内，避免了读读冲突。 |
| 示意图 |  |  |
| 示例代码 | ```cpp<br>uint64_t mask = 128;<br>UnaryRepeatParams params;<br>params.dstBlkStride = 1;<br>params.srcBlkStride = 16;<br>for(uint32_t i=0; i<16; i++) {<br>  AscendC::Adds(dstLocal[i * 128], srcLocal[i * 16], 0, mask, 1, params);<br>}<br>``` | ```cpp<br>uint64_t mask = 128;<br>UnaryRepeatParams params;<br>params.dstBlkStride = 8;<br>params.srcBlkStride = 1;<br>for(uint32_t i=0; i<8; i++) {<br>  AscendC::Adds(dstLocal[i * 16], srcLocal[i * 256], 0, mask, 2, params);<br>}<br>``` |

### 优化地址分配

实现连续 4096 个 float 元素的加法 z = x + y，通过在内存分配时适当扩大内存，保证在一个 Repeat 内，x 和 y 不会同时出现在同一个 bank group 内，x/y 和 z 不会同时出现同一个 bank 内。完整样例可参考避免 bank 冲突样例。实现方案对比如下：

| 实现方案 | 原始实现 | 优化实现 |
|----------|----------|----------|
| 实现方法 | 不做地址优化，直接使用 InitBuffer 分配内存，各个 Tensor 的地址分别为：<br>- x：起始地址 0x0，tensor 长度为 4096 * sizeof(float) 字节<br>- y：起始地址 0x4000，tensor 长度为 4096 * sizeof(float) 字节<br>- z：起始地址 0x8000，tensor 长度为 4096 * sizeof(float) 字节<br>在一个 Repeat 内，x 和 y 同时读同一个 bank group，x/y 和 z 同时读写同一个 bank。 | 优化地址，使用 InitBuffer 分配内存时适当扩大内存申请，各个 Tensor 的地址分别为：<br>- x：起始地址 0x0，tensor 长度为 (4096 * sizeof(float) + 256) 字节<br>- y：起始地址 0x4100，tensor 长度为 (64 * 1024 - (4096 * sizeof(float) + 256)) 字节<br>- z：起始地址 0x10000，tensor 长度为 4096 * sizeof(float) 字节<br>x 多申请 256 字节，避免一个 Repeat 内 x y 同时读同一个 bank group；y 多申请空间，确保 z 不会和 x/y 落入同一个 bank |
| 示意图 |  |  |
| 示例代码 | ```cpp<br>pipe.InitBuffer(inQueueX, 1, 4096 * sizeof(float));<br>pipe.InitBuffer(inQueueY, 1, 4096 * sizeof(float));<br>pipe.InitBuffer(outQueueZ, 1, 4096 * sizeof(float));<br>``` | ```cpp<br>pipe.InitBuffer(inQueueX, 1, 4096 * sizeof(float) + 256); // 多申请256字节<br>pipe.InitBuffer(inQueueY, 1, 64 * 1024 - (4096 * sizeof(float) + 256)); //多申请空间，确保z不会和x/y落入同一个bank， 64 * 1024是16个bank group的空间，4096 * sizeof(float) + 256是x所占的空间<br>pipe.InitBuffer(outQueueZ, 1, 4096 * sizeof(float));<br>``` |
