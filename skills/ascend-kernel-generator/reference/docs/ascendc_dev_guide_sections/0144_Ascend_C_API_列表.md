### Ascend C API 列表

Ascend C 提供一组类库 API，开发者使用标准 C++ 语法和类库 API 进行编程。Ascend C 编程类库 API 示意图如下所示，分为：

## Kernel API

用于实现算子核函数的 API 接口。包括：

- **基本数据结构**：kernel API 中使用到的基本数据结构，比如 GlobalTensor 和 LocalTensor。
- **基础 API**：实现对硬件能力的抽象，开放芯片的能力，保证完备性和兼容性。标注为 ISASI（Instruction Set Architecture Special Interface，硬件体系结构相关的接口）类别的 API，不能保证跨硬件版本兼容。
- **高阶 API**：实现一些常用的计算算法，用于提高编程开发效率，通常会调用多种基础 API 实现。高阶 API 包括数学库、Matmul、Softmax 等 API。高阶 API 可以保证兼容性。

## Host API

- **高阶 API 配套的 Tiling API**：kernel 侧高阶 API 配套的 Tiling API，方便开发者获取 kernel 计算时所需的 Tiling 参数。
- **Ascend C 算子原型注册与管理 API**：用于 Ascend C 算子原型定义和注册的 API。
- **Tiling 数据结构注册 API**：用于 Ascend C 算子 TilingData 数据结构定义和注册的 API。
- **平台信息获取 API**：在实现 Host 侧的 Tiling 函数时，可能需要获取一些硬件平台的信息，来支撑 Tiling 的计算，比如获取硬件平台的核数等信息。平台信息获取 API 提供获取这些平台信息的功能。

## 算子调测 API

用于算子调测的 API，包括孪生调试，性能调测等。

> 进行 Ascend C 算子 Host 侧编程时，需要使用基础数据结构和 API，请参考基础数据结构与接口；完成算子开发后，需要使用 Runtime API 完成算子的调用，请参考《应用开发指南 (C&C++)》中的 “acl API 参考” 章节。

---

## 基础 API

### 表 15-1 标量计算 API 列表

| 接口名 | 功能描述 |
|--------|----------|
| ScalarGetCountOfValue | 获取一个 uint64_t 类型数字的二进制中 0 或者 1 的个数。 |
| ScalarCountLeadingZero | 计算一个 uint64_t 类型数字前导 0 的个数（二进制从最高位到第一个 1 一共有多少个 0）。 |
| ScalarCast | 将一个 scalar 的类型转换为指定的类型。 |
| CountBitsCntSameAsSignBit | 计算一个 uint64_t 类型数字的二进制中，从最高数值位开始与符号位相同的连续比特位的个数。 |
| ScalarGetSFFValue | 获取一个 uint64_t 类型数字的二进制中第一个 0 或 1 出现的位置。 |
| ToBfloat16 | float 类型标量数据转换成 bfloat16_t 类型标量数据。 |
| ToFloat | bfloat16_t 类型标量数据转换成 float 类型标量数据。 |

### 表 15-2 矢量计算 API 列表

| 分类 | 接口名 | 功能描述 |
|------|--------|----------|
| 基础算术 | Exp | 按元素取自然指数。 |
| 基础算术 | Ln | 按元素取自然对数。 |
| 基础算术 | Abs | 按元素取绝对值。 |
| 基础算术 | Reciprocal | 按元素取倒数。 |
| 基础算术 | Sqrt | 按元素做开方。 |
| 基础算术 | Rsqrt | 按元素做开方后取倒数。 |
| 基础算术 | Relu | 按元素做线性整流 Relu。 |
| 基础算术 | Add | 按元素求和。 |
| 基础算术 | Sub | 按元素求差。 |
| 基础算术 | Mul | 按元素求积。 |
| 基础算术 | Div | 按元素求商。 |
| 基础算术 | Max | 按元素求最大值。 |
| 基础算术 | Min | 按元素求最小值。 |
| 基础算术 | Adds | 矢量内每个元素与标量求和。 |
| 基础算术 | Muls | 矢量内每个元素与标量求积。 |
| 基础算术 | Maxs | 源操作数矢量内每个元素与标量相比，如果比标量大，则取源操作数值，比标量的值小，则取标量值。 |
| 基础算术 | Mins | 源操作数矢量内每个元素与标量相比，如果比标量大，则取标量值，比标量的值小，则取源操作数值。 |
| 基础算术 | LeakyRelu | 按元素做带泄露线性整流 Leaky ReLU。 |
| 基础算术 | Axpy | 源操作数中每个元素与标量求积后和目的操作数中的对应元素相加。 |
| 逻辑计算 | Not | 按元素做按位取反。 |
| 逻辑计算 | And | 针对每对元素执行按位与运算。 |
| 逻辑计算 | Or | 针对每对元素执行按位或运算。 |
| 逻辑计算 | ShiftLeft | 对源操作数中的每个元素进行左移操作，左移的位数由输入参数 scalarValue 决定。 |
| 逻辑计算 | ShiftRight | 对源操作数中的每个元素进行右移操作，右移的位数由输入参数 scalarValue 决定。 |
| 复合计算 | AddRelu | 按元素求和，结果和 0 对比取较大值。 |
| 复合计算 | AddReluCast | 按元素求和，结果和 0 对比取较大值，并根据源操作数和目的操作数 Tensor 的数据类型进行精度转换。 |
| 复合计算 | AddDeqRelu | 依次计算按元素求和、结果进行 deq 量化后再进行 relu 计算（结果和 0 对比取较大值）。 |
| 复合计算 | SubRelu | 按元素求差，结果和 0 对比取较大值。 |
| 复合计算 | SubReluCast | 按元素求差，结果和 0 对比取较大值，并根据源操作数和目的操作数 Tensor 的数据类型进行精度转换。 |
| 复合计算 | MulAddDst | 按元素将 src0Local 和 src1Local 相乘并和 dstLocal 相加，将最终结果存放进 dstLocal 中。 |
| 复合计算 | MulCast | 按元素求积，并根据源操作数和目的操作数 Tensor 的数据类型进行精度转换。 |
| 复合计算 | FusedMulAdd | 按元素将 src0Local 和 dstLocal 相乘并加上 src1Local，最终结果存放入 dstLocal。 |
| 复合计算 | FusedMulAddRelu | 按元素将 src0Local 和 dstLocal 相乘并加上 src1Local，将结果和 0 作比较，取较大值，最终结果存放进 dstLocal 中。 |
| 比较指令 | Compare | 逐元素比较两个 tensor 大小，如果比较后的结果为真，则输出结果的对应比特位为 1，否则为 0。 |
| 比较指令 | Compare（结果存放寄存器） | 逐元素比较两个 tensor 大小，如果比较后的结果为真，则输出结果的对应比特位为 1，否则为 0。Compare 接口需要 mask 参数时，可以使用此接口。计算结果存放入寄存器中。 |
| 比较指令 | CompareScalar | 逐元素比较一个 tensor 中的元素和另一个 Scalar 的大小，如果比较后的结果为真，则输出结果的对应比特位为 1，否则为 0。 |
| 选择指令 | Select | 给定两个源操作数 src0 和 src1，根据 selMask（用于选择的 Mask 掩码）的比特位值选取元素，得到目的操作数 dst。选择的规则为：当 selMask 的比特位是 1 时，从 src0 中选取，比特位是 0 时从 src1 选取。 |
| 选择指令 | GatherMask | 以内置固定模式对应的二进制或者用户自定义输入的 Tensor 数值对应的二进制为 gather mask（数据收集的掩码），从源操作数中选取元素写入目的操作数中。 |
| 精度转换指令 | Cast | 根据源操作数和目的操作数 Tensor 的数据类型进行精度转换。 |
| 精度转换指令 | CastDeq | 对输入做量化并进行精度转换。 |
| 归约指令 | ReduceMax | 在所有的输入数据中找出最大值及最大值对应的索引位置。 |
| 归约指令 | ReduceMin | 在所有的输入数据中找出最小值及最小值对应的索引位置。 |
| 归约指令 | ReduceSum | 对所有的输入数据求和。 |
| 归约指令 | WholeReduceMax | 每个 repeat 内所有数据求最大值以及其索引 index。 |
| 归约指令 | WholeReduceMin | 每个 repeat 内所有数据求最小值以及其索引 index。 |
| 归约指令 | WholeReduceSum | 每个 repeat 内所有数据求和。 |
| 归约指令 | BlockReduceMax | 对每个 repeat 内所有元素求最大值。 |
| 归约指令 | BlockReduceMin | 对每个 repeat 内所有元素求最小值。 |
| 归约指令 | BlockReduceSum | 对每个 repeat 内所有元素求和。源操作数相加采用二叉树方式，两两相加。 |
| 归约指令 | PairReduceSum | 相邻两个（奇偶）元素求和。 |
| 归约指令 | RepeatReduceSum | 每个 repeat 内所有数据求和。和 WholeReduceSum 接口相比，不支持 mask 逐 bit 模式。建议使用功能更全面的 WholeReduceSum 接口。 |
| 数据转换 | Transpose | 可实现 16*16 的二维矩阵数据块的转置和 [N,C,H,W] 与 [N,H,W,C] 互相转换。 |
| 数据转换 | TransDataTo5HD | 数据格式转换，一般用于将 NCHW 格式转换成 NC1HWC0 格式。特别的，也可以用于二维矩阵数据块的转置。 |
| 数据填充 | Duplicate | 将一个变量或一个立即数，复制多次并填充到向量。 |
| 数据填充 | Brcb | 给定一个输入张量，每一次取输入张量中的 8 个数填充到结果张量的 8 个 datablock（32Bytes）中去，每个数对应一个 datablock。 |
| 数据填充 | CreateVecIndex | 以 firstValue 为起始值创建向量索引。 |
| 数据分散/数据收集 | Gather | 给定输入的张量和一个地址偏移张量，Gather 指令根据偏移地址将输入张量按元素收集到结果张量中。 |
| 掩码操作 | SetMaskCount | 设置 mask 模式为 Counter 模式。该模式下，不需要开发者去感知迭代次数、处理非对齐的尾块等操作，可直接传入计算数据量，实际迭代次数由 Vector 计算单元自动推断。 |
| 掩码操作 | SetMaskNorm | 设置 mask 模式为 Normal 模式。该模式为系统默认模式，支持开发者配置迭代次数。 |
| 掩码操作 | SetVectorMask | 用于在矢量计算时设置 mask。 |
| 掩码操作 | ResetMask | 恢复 mask 的值为默认值（全 1），表示矢量计算中每次迭代内的所有元素都将参与运算。 |
| 量化设置 | SetDeqScale | 设置 DEQSCALE 寄存器的值。 |

### 表 15-3 数据搬运 API 列表

| 接口名 | 功能描述 |
|--------|----------|
| DataCopy | 数据搬运接口，包括普通数据搬运、增强数据搬运、切片数据搬运、随路格式转换。 |
| Copy | VECIN、VECCALC、VECOUT 之间的搬运指令，支持 mask 操作和 DataBlock 间隔操作。 |

### 表 15-4 内存管理与同步控制 API 列表

| 接口名 | 功能描述 |
|--------|----------|
| TPipe | TPipe 是用来管理全局内存等资源的框架。通过 TPipe 类提供的接口可以完成内存等资源的分配管理操作。 |
| GetTPipePtr | 获取框架当前管理全局内存的 TPipe 指针，用户获取指针后，可进行 TPipe 相关的操作。 |
| TBufPool | TPipe 可以管理全局内存资源，而 TBufPool 可以手动管理或复用 Unified Buffer/L1 Buffer 物理内存，主要用于多个 stage 计算中 Unified Buffer/L1 Buffer 物理内存不足的场景。 |
| TQue | 提供入队出队等接口，通过队列（Queue）完成任务间同步。 |
| TQueBind | TQueBind 绑定源逻辑位置和目的逻辑位置，根据源位置和目的位置，来确定内存分配的位置、插入对应的同步事件，帮助开发者解决内存分配和管理、同步等问题。 |
| TBuf | 使用 Ascend C 编程的过程中，可能会用到一些临时变量。这些临时变量占用的内存可以使用 TBuf 数据结构来管理。 |
| InitSpmBuffer | 初始化 SPM Buffer。 |
| WriteSpmBuffer | 将需要溢出暂存的数据拷贝到 SPM Buffer 中。 |
| ReadSpmBuffer | 从 SPM Buffer 读回到 local 数据中。 |
| GetUserWorkspace | 获取用户使用的 workspace 指针。 |
| SetSysWorkSpace | 在进行融合算子编程时，由于框架通信机制需要使用到 workspace，也就是系统 workspace，所以在该场景下，开发者要调用该接口，设置系统 workspace 的指针。 |
| GetSysWorkSpacePtr | 获取系统 workspace 指针。 |
| TQueSync | TQueSync 类提供同步控制接口，开发者可以使用这类 API 来自行完成同步控制。 |
| IBSet | 当不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题时，通过调用该函数来插入同步语句来避免上述数据依赖时可能出现的数据读写错误问题。调用 IBSet 设置某一个核的标志位，与 IBWait 成对出现配合使用，表示核之间的同步等待指令，等待某一个核操作完成。 |
| IBWait | 当不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题时，通过调用该函数来插入同步语句来避免上述数据依赖时可能出现的数据读写错误问题。IBWait 与 IBSet 成对出现配合使用，表示核之间的同步等待指令，等待某一个核操作完成。 |
| SyncAll | 当不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题时，通过调用该函数来插入同步语句来避免上述数据依赖时可能出现的数据读写错误问题。目前多核同步分为硬同步和软同步，硬件同步是利用硬件自带的全核同步指令由硬件保证多核同步，软件同步是使用软件算法模拟实现。 |
| InitDetermineComputeWorkspace | 初始化 GM 共享内存的值，完成初始化后才可以调用 WaitPreBlock 和 NotifyNextBlock。 |
| WaitPreBlock | 通过读 GM 地址中的值，确认是否需要继续等待，当 GM 的值满足当前核的等待条件时，该核即可往下执行，进行下一步操作。 |
| NotifyNextBlock | 通过写 GM 地址，通知下一个核当前核的操作已完成，下一个核可以进行操作。 |
| SetNextTaskStart | 在 SuperKernel 的子 Kernel 中调用，调用后的指令可以和后续其他的子 Kernel 实现并行，提升整体性能。 |
| WaitPreTaskEnd | 在 SuperKernel 的子 Kernel 中调用，调用前的指令可以和前序其他的子 Kernel 实现并行，提升整体性能。 |

### 表 15-5 缓存处理 API 列表

| 接口名 | 功能描述 |
|--------|----------|
| DataCachePreload | 从源地址所在的特定 DDR 地址预加载数据到 data cache 中。 |
| DataCacheCleanAndInvalid | 该接口用来刷新 Cache，保证 Cache 的一致性。 |

### 表 15-6 系统变量访问 API 列表

| 接口名 | 功能描述 |
|--------|----------|
| GetBlockNum | 获取当前任务配置的 Block 数，用于代码内部的多核逻辑控制等。 |
| GetBlockIdx | 获取当前 core 的 index，用于代码内部的多核逻辑控制及多核偏移量计算等。 |
| GetDataBlockSizeInBytes | 获取当前芯片版本一个 datablock 的大小，单位为 byte。开发者根据 datablock 的大小来计算 API 指令中待传入的 repeatTime、DataBlock Stride、Repeat Stride 等参数值。 |
| GetArchVersion | 获取当前 AI 处理器架构版本号。 |
| GetTaskRation | 适用于 Cube/Vector 分离模式，用来获取 Cube/Vector 的配比。 |

### 表 15-7 原子操作接口列表

| 接口名 | 功能描述 |
|--------|----------|
| SetAtomicAdd | 设置接下来从 VECOUT 到 GM，L0C 到 GM，L1 到 GM 的数据传输是否进行原子累加，可根据参数不同设定不同的累加数据类型。 |
| SetAtomicType | 通过设置模板参数来设定原子操作不同的数据类型。 |
| SetAtomicNone | 原子操作函数，清空原子操作的状态。 |

### 表 15-8 Kernel Tiling 接口列表

| 接口名 | 功能描述 |
|--------|----------|
| GET_TILING_DATA | 用于获取算子 kernel 入口函数传入的 tiling 信息，并填入注册的 Tiling 结构体中，此函数会以宏展开的方式进行编译。如果用户注册了多个 TilingData 结构体，使用该接口返回默认注册的结构体。 |
| GET_TILING_DATA_WITH_STRUCT | 使用该接口指定结构体名称，可获取指定的 tiling 信息，并填入对应的 Tiling 结构体中，此函数会以宏展开的方式进行编译。 |
| GET_TILING_DATA_MEMBER | 用于获取 tiling 结构体的成员变量。 |
| TILING_KEY_IS | 在核函数中判断本次执行时的 tiling_key 是否等于某个 key，从而标识 tiling_key==key 的一条 kernel 分支。 |
| REGISTER_TILING_DEFAULT | 用于在 kernel 侧注册用户使用标准 C++ 语法自定义的默认 TilingData 结构体。 |
| REGISTER_TILING_FOR_TILINGKEY | 用于在 kernel 侧注册与 TilingKey 相匹配的 TilingData 自定义结构体；该接口需提供一个逻辑表达式，逻辑表达式以字符串 “TILING_KEY_VAR” 代指实际 TilingKey，表达 TilingKey 所满足的范围。 |
| KERNEL_TASK_TYPE_DEFAULT | 设置全局默认的 kernel type，对所有的 tiling key 生效。 |
| KERNEL_TASK_TYPE | 设置某一个具体的 tiling key 对应的 kernel type。 |

### 表 15-9 ISASI 接口列表

| 分类 | 接口名 | 功能描述 |
|------|--------|----------|
| 矢量计算 | VectorPadding | 根据 padMode（pad 模式）与 padSide（pad 方向）对源操作数按照 datablock 进行填充操作。 |
| 矢量计算 | BilinearInterpolation | 双线性插值操作，分为垂直迭代和水平迭代。 |
| 矢量计算 | GetCmpMask | 获取 Compare（结果存入寄存器）指令的比较结果。 |
| 矢量计算 | SetCmpMask | 为 Select 不传入 mask 参数的接口设置比较寄存器。 |
| 矢量计算 | GetAccVal | 获取 ReduceSum（针对 tensor 前 n 个数据计算）接口的计算结果。 |
| 矢量计算 | GetReduceMaxMinCount | 获取 ReduceMax、ReduceMin 连续场景下的最大/最小值以及相应的索引值。 |
| 矢量计算 | ProposalConcat | 将连续元素合入 Region Proposal 内对应位置，每次迭代会将 16 个连续元素合入到 16 个 Region Proposals 的对应位置里。 |
| 矢量计算 | ProposalExtract | 与 ProposalConcat 功能相反，从 Region Proposals 内将相应位置的单个元素抽取后重排，每次迭代处理 16 个 Region Proposals，抽取 16 个元素后连续排列。 |
| 矢量计算 | RpSort16 | 根据 Region Proposals 中的 score 域对其进行排序（score 大的排前面），每次排 16 个 Region Proposals。 |
| 矢量计算 | MrgSort4 | 将已经排好序的最多 4 条 region proposals 队列，排列并合并成 1 条队列，结果按照 score 域由大到小排序。 |
| 矢量计算 | Sort32 | 排序函数，一次迭代可以完成 32 个数的排序。 |
| 矢量计算 | MrgSort | 将已经排好序的最多 4 条队列，合并排列成 1 条队列，结果按照 score 域由大到小排序。 |
| 矢量计算 | GetMrgSortResult | 获取 MrgSort 或 MrgSort4 已经处理过的队列里的 Region Proposal 个数，并依次存储在四个 List 入参中。 |
| 矢量计算 | Gatherb | 给定一个输入的张量和一个地址偏移张量，Gatherb 指令根据偏移地址将输入张量收集到结果张量中。 |
| 矢量计算 | Scatter | 给定一个连续的输入张量和一个目的地址偏移张量，Scatter 指令根据偏移地址生成新的结果张量后将输入张量分散到结果张量中。 |
| 矩阵计算 | Mmad | 完成矩阵乘加操作。 |
| 矩阵计算 | MmadWithSparse | 完成矩阵乘加操作，传入的左矩阵 A 为稀疏矩阵，右矩阵 B 为稠密矩阵。 |
| 矩阵计算 | SetHF32Mode | 此接口同 15.1.4.8.5 SetHF32TransMode 与 15.1.4.8.3 SetMMLayoutTransform 一样，都用于设置寄存器的值。SetHF32Mode 接口用于设置 MMAD 的 HF32 模式。 |
| 矩阵计算 | SetHF32TransMode | 此接口同 15.1.4.8.4 SetHF32Mode 与 15.1.4.8.3 SetMMLayoutTransform 一样，都用于设置寄存器的值。SetHF32TransMode 用于设置 MMAD 的 HF32 取整模式，仅在 MMAD 的 HF32 模式生效时有效。 |
| 矩阵计算 | SetMMLayoutTransform | 此接口同 SetHF32Mode 与 SetHF32TransMode 一样，都用于设置寄存器的值，其中 SetMMLayoutTransform 接口用于设置 MMAD 的 M/N 方向。 |
| 矩阵计算 | Conv2D | 计算给定输入张量和权重张量的 2-D 卷积，输出结果张量。Conv2d 卷积层多用于图像识别，使用过滤器提取图像中的特征。 |
| 矩阵计算 | Gemm | 根据输入的切分规则，将给定的两个输入张量做矩阵乘，输出至结果张量。将 A 和 B 两个输入矩阵乘法在一起，得到一个输出矩阵 C。 |
| 数据搬运 | DataCopyPad | 该接口提供数据非对齐搬运的功能。 |
| 数据搬运 | SetPadValue | 设置 DataCopyPad 接口填充的数值。 |
| 数据搬运 | SetFixPipeConfig | DataCopy（CO1->GM、CO1->A1）过程中进行随路量化时，通过调用该接口设置量化流程中 tensor 量化参数。 |
| 数据搬运 | SetFixpipeNz2ndFlag | DataCopy（CO1->GM、CO1->A1）过程中进行随路格式转换（NZ2ND）时，通过调用该接口设置 NZ2ND 相关配置。 |
| 数据搬运 | SetFixpipePreQuantFlag | DataCopy（CO1->GM、CO1->A1）过程中进行随路量化时，通过调用该接口设置量化流程中 scalar 量化参数。 |
| 数据搬运 | SetFixPipeClipRelu | DataCopy（CO1->GM）过程中进行随路量化后，通过调用该接口设置 ClipRelu 操作的最大值。 |
| 数据搬运 | SetFixPipeAddr | DataCopy（CO1->GM）过程中进行随路量化后，通过调用该接口设置 element-wise 操作时 LocalTensor 的地址。 |
| 数据搬运 | InitConstValue | 初始化 LocalTensor（TPosition 为 A1/A2/B1/B2）为某一个具体的数值。 |
| 数据搬运 | LoadData | LoadData 包括 Load2D 和 Load3D 数据加载功能。 |
| 数据搬运 | LoadDataWithTranspose | 该接口实现带转置的 2D 格式数据从 A1/B1 到 A2/B2 的加载。 |
| 数据搬运 | SetAippFunctions | 设置图片预处理（AIPP，AI core pre-process）相关参数。 |
| 数据搬运 | LoadImageToLocal | 将图像数据从 GM 搬运到 A1/B1。搬运过程中可以完成图像预处理操作：包括图像翻转，改变图像尺寸（抠图，裁边，缩放，伸展），以及色域转换，类型转换等。 |
| 数据搬运 | LoadUnZipIndex | 加载 GM 上的压缩索引表到内部寄存器。 |
| 数据搬运 | LoadDataUnzip | 将 GM 上的数据解压并搬运到 A1/B1/B2 上。 |
| 数据搬运 | LoadDataWithSparse | 用于搬运存放在 B1 里的 512B 的稠密权重矩阵到 B2 里，同时读取 128B 的索引矩阵用于稠密矩阵的稀疏化。 |
| 数据搬运 | SetFmatrix | 用于调用 Load3Dv1/Load3Dv2 时设置 FeatureMap 的属性描述。 |
| 数据搬运 | SetLoadDataBoundary | 设置 Load3D 时 A1/B1 边界值。 |
| 数据搬运 | SetLoadDataRepeat | 用于设置 Load3Dv2 接口的 repeat 参数。设置 repeat 参数后，可以通过调用一次 Load3Dv2 接口完成多个迭代的数据搬运。 |
| 数据搬运 | SetLoadDataPaddingValue | 设置 padValue，用于 Load3Dv1/Load3Dv2。 |
| 数据搬运 | Fixpipe | 矩阵计算完成后，对结果进行处理，例如对计算结果进行量化操作，并把数据从 CO1 搬迁到 Global Memory 中。 |
| 同步控制 | SetFlag/WaitFlag | 同一核内不同流水线之间的同步指令。具有数据依赖的不同流水指令之间需要插此同步。 |
| 同步控制 | PipeBarrier | 阻塞相同流水，具有数据依赖的相同流水之间需要插此同步。 |
| 同步控制 | DataSyncBarrier | 用于阻塞后续的指令执行，直到所有之前的内存访问指令（需要等待的内存位置可通过参数控制）执行结束。 |
| 同步控制 | CrossCoreSetFlag | 针对分离模式，AI Core 上的 Cube 核（AIC）与 Vector 核（AIV）之间的同步设置指令。 |
| 同步控制 | CrossCoreWaitFlag | 针对分离模式，AI Core 上的 Cube 核（AIC）与 Vector 核（AIV）之间的同步等待指令。 |
| 缓存处理 | ICachePreLoad | 从指令所在 DDR 地址预加载指令到 ICache 中。 |
| 缓存处理 | GetICachePreloadStatus | 获取 ICACHE 的 PreLoad 的状态。 |
| 系统变量访问 | GetProgramCounter | 获取程序计数器的指针，程序计数器用于记录当前程序执行的位置。 |
| 系统变量访问 | GetSubBlockNum | 获取 AI Core 上 Vector 核的数量。 |
| 系统变量访问 | GetSubBlockIdx | 获取 AI Core 上 Vector 核的 ID。 |
| 系统变量访问 | GetSystemCycle | 获取当前系统 cycle 数，若换算成时间需要按照 50MHz 的频率，时间单位为 us，换算公式为：time = (cycle数/50) us。 |
| 系统变量访问 | CheckLocalMemoryIA | 监视设定范围内的 UB 读写行为，如果监视到有设定范围的读写行为则会出现 EXCEPTION 报错，未监视到设定范围的读写行为则不会报错。 |
| 原子操作 | SetAtomicMax | 原子操作函数，设置后续从 VECOUT 传输到 GM 的数据是否执行原子比较，将待拷贝的内容和 GM 已有内容进行比较，将最大值写入 GM。 |
| 原子操作 | SetAtomicMin | 原子操作函数，设置后续从 VECOUT 传输到 GM 的数据是否执行原子比较，将待拷贝的内容和 GM 已有内容进行比较，将最小值写入 GM。 |
| 原子操作 | SetStoreAtomicConfig | 设置原子操作使能位与原子操作类型。 |
| 原子操作 | GetStoreAtomicConfig | 获取原子操作使能位与原子操作类型的值。 |
| 资源管理 | CubeResGroupHandle | CubeResGroupHandle 用于在分离模式下通过软同步控制 AIC 和 AIV 之间进行通讯，实现 AI Core 计算资源分组。 |
| 资源管理 | GroupBarrier | 当同一个 CubeResGroupHandle 中的两个 AIV 任务之间存在依赖关系时，可以使用 GroupBarrier 控制同步。 |
| 资源管理 | KfcWorkspace | KfcWorkspace 为通信空间描述符，管理不同 CubeResGroupHandle 的消息通信区划分，与 CubeResGroupHandle 配合使用。KfcWorkspace 的构造函数用于创建 KfcWorkspace 对象。 |

---

## 高阶 API

### 表 15-10 数学库 API 列表

| 接口名 | 功能描述 |
|--------|----------|
| Acos | 按元素做反余弦函数计算。 |
| Acosh | 按元素做双曲反余弦函数计算。 |
| Asin | 按元素做反正弦函数计算。 |
| Asinh | 按元素做反双曲正弦函数计算。 |
| Atan | 按元素做三角函数反正切运算。 |
| Atanh | 按元素做反双曲正切余弦函数计算。 |
| Axpy | 源操作数中每个元素与标量求积后和目的操作数中的对应元素相加。 |
| Ceil | 获取大于或等于 x 的最小的整数值，即向正无穷取整操作。 |
| ClampMax | 将 srcTensor 中大于 scalar 的数替换为 scalar，小于等于 scalar 的数保持不变，作为 dstTensor 输出。 |
| ClampMin | 将 srcTensor 中小于 scalar 的数替换为 scalar，大于等于 scalar 的数保持不变，作为 dstTensor 输出。 |
| Cos | 按元素做三角函数余弦运算。 |
| Cosh | 按元素做双曲余弦函数计算。 |
| CumSum | 对数据按行依次累加或按列依次累加。 |
| Digamma | 按元素计算 x 的 gamma 函数的对数导数。 |
| Erf | 按元素做误差函数计算，也称为高斯误差函数。 |
| Erfc | 返回输入 x 的互补误差函数结果，积分区间为 x 到无穷大。 |
| Exp | 按元素取自然指数。 |
| Floor | 获取小于或等于 x 的最小的整数值，即向负无穷取整操作。 |
| Fmod | 按元素计算两个浮点数相除后的余数。 |
| Frac | 按元素做取小数计算。 |
| Lgamma | 按元素计算 x 的 gamma 函数的绝对值并求自然对数。 |
| Log | 按元素以 e、2、10 为底做对数运算。 |
| Power | 实现按元素做幂运算功能。 |
| Round | 将输入的元素四舍五入到最接近的整数。 |
| Sign | 按元素执行 Sign 操作，Sign 是指返回输入数据的符号。 |
| Sin | 按元素做正弦函数计算。 |
| Sinh | 按元素做双曲正弦函数计算。 |
| Tan | 按元素做正切函数计算。 |
| Tanh | 按元素做逻辑回归 Tanh。 |
| Trunc | 按元素做浮点数截断操作，即向零取整操作。 |
| Xor | 按元素执行 Xor（异或）运算。 |

### 表 15-11 量化反量化 API 列表

| 接口名 | 功能描述 |
|--------|----------|
| AscendAntiQuant | 按元素做伪量化计算，比如将 int8_t 数据类型伪量化为 half 数据类型。 |
| AscendDequant | 按元素做反量化计算，比如将 int32_t 数据类型反量化为 half/float 等数据类型。 |
| AscendQuant | 按元素做量化计算，比如将 half/float 数据类型量化为 int8_t 数据类型。 |

### 表 15-12 数据归一化 API 列表

| 接口名 | 功能描述 |
|--------|----------|
| BatchNorm | 对于每个 batch 中的样本，对其输入的每个特征在 batch 的维度上进行归一化。 |
| DeepNorm | 在深层神经网络训练过程中，可以替代 LayerNorm 的一种归一化方法。 |
| GroupNorm | 将输入的 C 维度分为 groupNum 组，对每一组数据进行标准化。 |
| LayerNorm | 将输入数据收敛到 [0, 1] 之间，可以规范网络层输入输出数据分布的一种归一化方法。 |
| LayerNormGrad | 用于计算 LayerNorm 的反向传播梯度。 |
| LayerNormGradBeta | 用于获取反向 beta/gmma 的数值，和 LayerNormGrad 共同输出 pdx, gmma 和 beta。 |
| Normalize | LayerNorm 中，已知均值和方差，计算 shape 为 [A，R] 的输入数据的标准差的倒数 rstd 和归一化输出 y。 |
| RmsNorm | 实现对 shape 大小为 [B，S，H] 的输入数据的 RmsNorm 归一化。 |
| WelfordUpdate | 实现 Welford 算法的前处理。 |
| WelfordFinalize | 实现 Welford 算法的后处理。 |

### 表 15-13 激活函数 API 列表

| 接口名 | 功能描述 |
|--------|----------|
| AdjustSoftMaxRes | 用于对 SoftMax 相关计算结果做后处理，调整 SoftMax 的计算结果为指定的值。 |
| FasterGelu | FastGelu 化简版本的一种激活函数。 |
| FasterGeluV2 | 实现 FastGeluV2 版本的一种激活函数。 |
| GeGLU | 采用 GeLU 作为激活函数的 GLU 变体。 |
| Gelu | GELU 是一个重要的激活函数，其灵感来源于 relu 和 dropout，在激活中引入了随机正则的思想。 |
| LogSoftMax | 对输入 tensor 做 LogSoftmax 计算。 |
| ReGlu | 一种 GLU 变体，使用 Relu 作为激活函数。 |
| Sigmoid | 按元素做逻辑回归 Sigmoid。 |
| Silu | 按元素做 Silu 运算。 |
| SimpleSoftMax | 使用计算好的 sum 和 max 数据对输入 tensor 做 softmax 计算。 |
| SoftMax | 对输入 tensor 按行做 Softmax 计算。 |
| SoftmaxFlash | SoftMax 增强版本，除了可以对输入 tensor 做 softmaxflash 计算，还可以根据上一次 softmax 计算的 sum 和 max 来更新本次的 softmax 计算结果。 |
| SoftmaxFlashV2 | SoftmaxFlash 增强版本，对应 FlashAttention-2 算法。 |
| SoftmaxFlashV3 | SoftmaxFlash 增强版本，对应 Softmax PASA 算法。 |
| SoftmaxGrad | 对输入 tensor 做 grad 反向计算的一种方法。 |
| SoftmaxGradFront | 对输入 tensor 做 grad 反向计算的一种方法。 |
| SwiGLU | 采用 Swish 作为激活函数的 GLU 变体。 |
| Swish | 神经网络中的 Swish 激活函数。 |

### 表 15-14 归约操作 API 列表

| 接口名 | 功能描述 |
|--------|----------|
| Sum | 获取最后一个
