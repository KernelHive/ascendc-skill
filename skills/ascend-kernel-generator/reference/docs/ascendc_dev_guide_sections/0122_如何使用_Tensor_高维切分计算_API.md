## 如何使用 Tensor 高维切分计算 API

## 说明

- 本章节对矢量计算基础 API 中的 tensor 高维切分计算接口做解释说明。如果您不需要使用此类接口，可略过该章节。
- 下文中的 `repeatTime`、`dataBlockStride`、`repeatStride`、`mask` 为通用描述，其命名不一定与具体指令中的参数命名完全对应。
  例如，单次迭代内不同 datablock 间地址步长 `dataBlockStride` 参数，在单目 API 中，对应为 `dstBlkStride`、`srcBlkStride` 参数；在双目 API 中，对应为 `dstBlkStride`、`src0BlkStride`、`src1BlkStride` 参数。
  您可以在具体接口的参数说明中，找到参数含义的描述。

使用 tensor 高维切分计算 API 可充分发挥硬件优势，支持开发者控制指令的迭代执行和操作数的地址间隔，功能更加灵活。

矢量计算通过 Vector 计算单元完成，矢量计算的源操作数和目的操作数均通过 Unified Buffer（UB）来进行存储。Vector 计算单元每个迭代会从 UB 中取出 8 个 datablock（每个 datablock 数据块内部地址连续，长度 32Byte），进行计算，并写入对应的 8 个 datablock 中。下图为单次迭代内的 8 个 datablock 进行 Exp 计算的示意图。

![图 12-1 单次迭代内的 8 个 datablock 进行 Exp 计算示意图]()

- 矢量计算 API 支持开发者通过 `repeatTime` 来配置迭代次数，从而控制指令的多次迭代执行。假设 `repeatTime` 设置为 2，矢量计算单元会进行 2 个迭代的计算，可计算出 `2 * 8（每个迭代 8 个 datablock） * 32Byte（每个 datablock 32Byte） = 512Byte` 的结果。如果数据类型为 half，则计算了 256 个元素。下图展示了 2 次迭代 Exp 计算的示意图。由于硬件限制，`repeatTime` 不能超过 255。

![图 12-2 2 次迭代 Exp 计算]()

- 针对同一个迭代中的数据，可以通过 `mask` 参数进行掩码操作来控制实际参与计算的个数。下图为进行 Abs 计算时通过 mask 逐比特模式按位控制哪些元素参与计算的示意图，1 表示参与计算，0 表示不参与计算。

![图 12-3 通过 mask 参数进行掩码操作示意图（以 float 数据类型为例）]()

- 矢量计算单元还支持带间隔的向量计算，通过 `dataBlockStride`（单次迭代内不同 datablock 间地址步长）和 `repeatStride`（相邻迭代间相同 datablock 的地址步长）来进行配置。

  - **dataBlockStride**  
    如果需要控制单次迭代内，数据处理的步长，可以通过设置同一迭代内不同 datablock 的地址步长 `dataBlockStride` 来实现。下图给出了单次迭代内非连续场景的示意图，示例中源操作数的 `dataBlockStride` 配置为 2，表示单次迭代内不同 datablock 间地址步长（起始地址之间的间隔）为 2 个 datablock。

    ![图 12-4 单次迭代内非连续场景的示意图]()

  - **repeatStride**  
    当 `repeatTime` 大于 1，需要多次迭代完成矢量计算时，您可以根据不同的使用场景合理设置相邻迭代间相同 datablock 的地址步长 `repeatStride` 的值。  
    下图给出了多次迭代间非连续场景的示意图，示例中源操作数和目的操作数的 `repeatStride` 均配置为 9，表示相邻迭代间相同 datablock 起始地址之间的间隔为 9 个 datablock。相同 datablock 是指 datablock 在迭代内的位置相同，比如下图中的 src1 和 src9 处于相邻迭代，在迭代内都是第一个 datablock 的位置，其间隔即为 `repeatStride` 的数值。

    ![图 12-5 多次迭代间非连续场景的示意图]()

下文中给出了 `dataBlockStride`、`repeatStride`、`mask` 的详细配置说明和示例。

## dataBlockStride

`dataBlockStride` 是指同一迭代内不同 datablock 的地址步长。

- **连续计算**：`dataBlockStride` 设置为 1，对同一迭代内的 8 个 datablock 数据连续进行处理。
- **非连续计算**：`dataBlockStride` 值大于 1（如取 2），同一迭代内不同 datablock 之间在读取数据时出现一个 datablock 的间隔，如下图所示。

![图 12-6 dataBlockStride 不同取值举例]()

## repeatStride

`repeatStride` 是指相邻迭代间相同 datablock 的地址步长。

- **连续计算场景**：假设定义一个 Tensor 供目的操作数和源操作数同时使用（即地址重叠），`repeatStride` 取值为 8。此时，矢量计算单元第一次迭代读取连续 8 个 datablock，第二轮迭代读取下一个连续的 8 个 datablock，通过多次迭代即可完成所有输入数据的计算。
- **非连续计算场景**：`repeatStride` 取值大于 8（如取 10）时，则相邻迭代间矢量计算单元读取的数据在地址上不连续，出现 2 个 datablock 的间隔。
- **反复计算场景**：`repeatStride` 取值为 0 时，矢量计算单元会对首个连续的 8 个 datablock 进行反复读取和计算。
- **部分重复计算**：`repeatStride` 取值大于 0 且小于 8 时，相邻迭代间部分数据会被矢量计算单元重复读取和计算，此种情形一般场景不涉及。

## mask 参数

`mask` 用于控制每次迭代内参与计算的元素。可通过连续模式和逐 bit 模式两种方式进行设置。

### 连续模式

表示前面连续的多少个元素参与计算。数据类型为 `uint64_t`。取值范围和源操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同（当前数据类型单次迭代时能处理的元素个数最大值为：`256 / sizeof(数据类型)`）。

- 当操作数的数据类型占 bit 位 16 位时（如 `half`/`uint16_t`），`mask ∈ [1, 128]`
- 当操作数为 32 位时（如 `float`/`int32_t`），`mask ∈ [1, 64]`

具体样例如下：

```cpp
// int16_t 数据类型单次迭代能处理的元素个数最大值为 256/sizeof(int16_t) = 128，mask = 64，mask ∈ [1, 128]，所以是合法输入
// repeatTime = 1, 共 128 个元素，单次迭代能处理 128 个元素，故 repeatTime = 1
// dstBlkStride, src0BlkStride, src1BlkStride = 1, 单次迭代内连续读取和写入数据
// dstRepStride, src0RepStride, src1RepStride = 8, 迭代间的数据连续读取和写入
uint64_t mask = 64;
AscendC::Add(dstLocal, src0Local, src1Local, mask, 1, { 1, 1, 1, 8, 8, 8 });
```

结果示例如下：

```
输入数据(src0Local): [1 2 3 ... 64 ...128]
输入数据(src1Local): [1 2 3 ... 64 ...128]
输出数据(dstLocal): [2 4 6 ... 128 undefined...undefined]
```

```cpp
// int32_t 数据类型单次迭代能处理的元素个数最大值为 256/sizeof(int32_t) = 64，mask = 64，mask ∈ [1, 64]，所以是合法输入
// repeatTime = 1, 共 64 个元素，单次迭代能处理 64 个元素，故 repeatTime = 1
// dstBlkStride, src0BlkStride, src1BlkStride = 1, 单次迭代内连续读取和写入数据
// dstRepStride, src0RepStride, src1RepStride = 8, 迭代间的数据连续读取和写入
uint64_t mask = 64;
AscendC::Add(dstLocal, src0Local, src1Local, mask, 1, { 1, 1, 1, 8, 8, 8 });
```

结果示例如下：

```
输入数据(src0Local): [1 2 3 ... 64]
输入数据(src1Local): [1 2 3 ... 64]
输出数据(dstLocal): [2 4 6 ... 128]
```

### 逐 bit 模式

可以按位控制哪些元素参与计算，bit 位的值为 1 表示参与计算，0 表示不参与。

`mask` 为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关：

- 当操作数为 16 位时，数组长度为 2，`mask[0]`、`mask[1] ∈ [0, 2^64-1]` 并且不同时为 0
- 当操作数为 32 位时，数组长度为 1，`mask[0] ∈ (0, 2^64-1]`
- 当操作数为 64 位时，数组长度为 1，`mask[0] ∈ (0, 2^32-1]`

具体样例如下：

```cpp
// 数据类型为 int16_t
uint64_t mask[2] = {6148914691236517205, 6148914691236517205};
// repeatTime = 1, 共 128 个元素，单次迭代能处理 128 个元素，故 repeatTime = 1
// dstBlkStride, src0BlkStride, src1BlkStride = 1, 单次迭代内连续读取和写入数据
// dstRepStride, src0RepStride, src1RepStride = 8, 迭代间的数据连续读取和写入
AscendC::Add(dstLocal, src0Local, src1Local, mask, 1, { 1, 1, 1, 8, 8, 8 });
```

结果示例如下：

```
输入数据(src0Local): [1 2 3 ... 64 ...127 128]
输入数据(src1Local): [1 2 3 ... 64 ...127 128]
输出数据(dstLocal): [2 undefined 6 ... undefined ...254 undefined]
```

mask 过程如下：

`mask = {6148914691236517205, 6148914691236517205}`（注：6148914691236517205 表示 64 位二进制数 `0b010101....01`，mask 按照低位到高位的顺序排布）

```cpp
// 数据类型为 int32_t
uint64_t mask[1] = {6148914691236517205};
// repeatTime = 1, 共 64 个元素，单次迭代能处理 64 个元素，故 repeatTime = 1
// dstBlkStride, src0BlkStride, src1BlkStride = 1, 单次迭代内连续读取和写入数据
// dstRepStride, src0RepStride, src1RepStride = 8, 迭代间的数据连续读取和写入
AscendC::Add(dstLocal, src0Local, src1Local, mask, 1, { 1, 1, 1, 8, 8, 8 });
```

结果示例如下：

```
输入数据(src0Local): [1 2 3 ... 63 64]
输入数据(src1Local): [1 2 3 ... 63 64]
输出数据(dstLocal): [2 undefined 6 ... 126 undefined]
```

mask 过程如下：

`mask = {6148914691236517205, 0}`（注：6148914691236517205 表示 64 位二进制数 `0b010101....01`）
