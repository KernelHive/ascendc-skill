###### CalcTschBlockDim

## 功能说明

针对 Cube、Vector 分离模式，用于计算 Cube、Vector 融合算子的 blockDim。针对 Vector/Cube 融合计算的算子，启动时，按照 AIV 和 AIC 组合启动，blockDim 用于设置启动多少个组合执行。

例如：某款 AI 处理器上有 40 个 Vector 核 + 20 个 Cube 核，一个组合是 2 个 Vector 和 1 个 Cube 核，建议设置为 20，此时会启动 20 个组合，即 40 个 Vector 和 20 个 Cube 核。使用该接口可以自动获取合适的 blockDim 值。

获取该值后，使用 `SetBlockDim` 进行 blockDim 的设置。

## 函数原型

```cpp
uint32_t CalcTschBlockDim(uint32_t sliceNum, uint32_t aicCoreNum, uint32_t aivCoreNum) const
```

## 参数说明

| 参数        | 输入/输出 | 说明                                                                 |
|-------------|-----------|----------------------------------------------------------------------|
| sliceNum    | 输入      | 数据切分的份数                                                       |
| aicCoreNum  | 输入      | 如果算子实现使用了矩阵计算 API，请传入 `GetCoreNumAic` 返回的数量，否则传入 0 |
| aivCoreNum  | 输入      | 如果算子实现使用了矢量计算 API，请传入 `GetCoreNumAiv` 返回的数量，否则传入 0 |

## 返回值说明

返回用于底层任务调度的核数。

## 约束说明

无

## 调用示例

```cpp
ge::graphStatus TilingXXX(gert::TilingContext* context) {
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  auto aicNum = ascendcPlatform.GetCoreNumAic();
  auto aivNum = ascendcPlatform.GetCoreNumAiv();
  // 按照 aivNum 切分数据，并进行计算
  uint32_t sliceNum = aivNum;
  context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(sliceNum, aicNum, aivNum));
  return ret;
}
```
