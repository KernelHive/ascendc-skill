### Vector 算子灵活运用 Counter 模式

【优先级】高

## 描述

Normal 模式下，通过迭代次数 `repeatTimes` 和掩码 `mask`，控制 Vector 算子中矢量计算 API 的计算数据量。当用户想要指定 API 计算的总元素个数时，首先需要自行判断是否存在不同的主块和尾块：

- 主块需要将 `mask` 设置为全部元素参与计算，并计算主块所需迭代次数
- 根据尾块中剩余元素个数重置 `mask`，再进行尾块的运算

这中间涉及大量 Scalar 计算。

Counter 模式下，用户不需要计算迭代次数以及判断是否存在尾块。将 mask 模式设置为 Counter 模式后，只需要设置 `mask` 为 `{0, 总元素个数}`，然后调用相应的 API。处理逻辑更简便，减少了指令数量和 Scalar 计算量，同时更加高效地利用了指令单次执行的并发能力，进而提升性能。

> 提示：Normal 模式和 Counter 模式、掩码的介绍可参考《Ascend C 算子开发指南》中的“常用操作 > 如何使用掩码操作 API”章节。

以下反例和正例中的代码均以 AddCustom 算子为例，修改其中 Add 接口的调用代码，以说明 Counter 模式的优势。

```cpp
AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);
```

## 反例

输入数据类型为 `half` 的 `xLocal`、`yLocal`，数据量均为 15000。Normal 模式下，每个迭代内参与计算的元素个数最多为 `256B / sizeof(half) = 128` 个，所以 15000 次 Add 计算会被分为：

- 主块计算：`15000 / 128 = 117` 次迭代，每次迭代 128 个元素参与计算
- 尾块计算：1 次迭代，该迭代 `15000 - 117 * 128 = 24` 个元素参与计算

从代码角度，需要计算主块的 `repeatTimes`、尾块元素个数；主块计算时，设置 mask 值为 128，尾块计算时，需要设置 mask 值为尾块元素个数 24；这些过程均涉及 Scalar 计算。

```cpp
uint32_t ELE_SIZE = 15000;
AscendC::BinaryRepeatParams binaryParams;

uint32_t numPerRepeat = 256 / sizeof(DTYPE_X); // DTYPE_X 为 half 数据类型
uint32_t mainRepeatTimes = ELE_SIZE / numPerRepeat;
uint32_t tailEleNum = ELE_SIZE % numPerRepeat;

AscendC::SetMaskNorm();
AscendC::SetVectorMask<DTYPE_X, AscendC::MaskMode::NORMAL>(numPerRepeat); // 设置 normal 模式 mask，使每个迭代计算 128 个数
AscendC::Add<DTYPE_X, false>(zLocal, xLocal, yLocal, AscendC::MASK_PLACEHOLDER, mainRepeatTimes, binaryParams); // MASK_PLACEHOLDER 值为 0，此处为 mask 占位，实际 mask 值以 SetVectorMask 设置的为准
if (tailEleNum > 0) {
    AscendC::SetVectorMask<DTYPE_X, AscendC::MaskMode::NORMAL>(tailEleNum); // 设置 normal 模式 mask，使每个迭代计算 24 个数
    // 偏移 tensor 的起始地址，在 xLocal 和 yLocal 的 14976 个元素处，进行尾块计算
    AscendC::Add<DTYPE_X, false>(zLocal[mainRepeatTimes * numPerRepeat], xLocal[mainRepeatTimes * numPerRepeat], yLocal[mainRepeatTimes * numPerRepeat], AscendC::MASK_PLACEHOLDER, 1, binaryParams);
}
AscendC::ResetMask(); // 还原 mask 值
```

## 正例

输入数据类型为 `half` 的 `xLocal`、`yLocal`，数据量均为 15000。Counter 模式下，只需要设置 mask 为所有参与计算的元素个数 15000，然后直接调用 Add 指令，即可完成所有计算，不需要繁琐的主尾块计算，代码较为简练。

当要处理多达 15000 个元素的矢量计算时，Counter 模式的优势更明显，不需要反复修改主块和尾块不同的 mask 值，减少了指令条数以及 Scalar 计算量，并充分利用了指令单次执行的并发能力。

```cpp
uint32_t ELE_SIZE = 15000;
AscendC::BinaryRepeatParams binaryParams;
AscendC::SetMaskCount();
AscendC::SetVectorMask<DTYPE_X, AscendC::MaskMode::COUNTER>(ELE_SIZE); // 设置 counter 模式 mask，总共计算 15000 个数
AscendC::Add<DTYPE_X, false>(zLocal, xLocal, yLocal, AscendC::MASK_PLACEHOLDER, 1, binaryParams); // MASK_PLACEHOLDER 值为 0，此处为 mask 占位，实际 mask 值以 SetVectorMask 设置的为准
AscendC::ResetMask(); // 还原 mask 值
```

## 性能对比

- 图 5-21 Normal 模式和 Counter 模式下的 Scalar 执行时间对比
- 图 5-22 Normal 模式和 Counter 模式下的 Vector 执行时间对比

以上性能数据是分别循环运行 1000 次反例和正例代码得到的 Scalar 和 Vector 执行时间。从上述两幅性能对比图和示例代码可以看到，使用 Counter 模式能够大幅度简化代码，易于维护，同时能够降低 Scalar 和 Vector 计算耗时，获得性能提升。
