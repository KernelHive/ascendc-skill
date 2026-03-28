##### GetPlatformInfo

## 函数功能
获取 PlatformInfos 指针。PlatformInfos 类用于管理和查询硬件平台的相关信息，包括核数、内存带宽等。

## 函数原型
```cpp
fe::PlatFormInfos *GetPlatformInfo() const
```

## 参数说明
无。

## 返回值说明
PlatformInfos 指针承载的是内部的硬件平台信息，这个信息作为 PlatformAscendC 的输入，开发者通过 PlatformAscendC 提供的方法获取硬件平台信息。

## 约束说明
无。

## 调用示例
```cpp
ge::graphStatus TilingXXX(gert::TilingContext* context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto coreNum = ascendcPlatform.GetCoreNum();
    // ... 根据核数自行设计 Tiling 策略
    context->SetBlockDim(coreNum);
    return ret;
}
```
