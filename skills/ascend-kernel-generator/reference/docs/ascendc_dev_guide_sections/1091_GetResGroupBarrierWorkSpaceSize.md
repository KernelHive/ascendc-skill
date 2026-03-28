###### GetResGroupBarrierWorkSpaceSize

## 功能说明

获取 GroupBarrier 所需要的 workspace 空间大小。

## 函数原型

```cpp
uint32_t GetResGroupBarrierWorkSpaceSize(void) const
```

## 参数说明

无

## 返回值说明

当前 GroupBarrier 所需要的 workspace 大小。

## 约束说明

无

## 调用示例

```cpp
// 用户自定义的 tiling 函数
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AddApiTiling tiling;
    ...
    // 如需要使用系统 workspace 需要调用 GetLibApiWorkSpaceSize 获取系统 workspace 的大小。
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    
    // 设置用户需要使用的 workspace 和 GroupBarrier 需要的大小作为 usrWorkspace 的总大小。
    size_t usrSize = 256 + ascendcPlatform.GetResGroupBarrierWorkSpaceSize(); // 设置用户需要使用的 workspace 大小。
    size_t *currentWorkspace = context->GetWorkspaceSizes(1); // 通过框架获取 workspace 的指针，GetWorkspaceSizes 入参为所需 workspace 的块数。当前限制使用一块。
    currentWorkspace[0] = usrSize + sysWorkspaceSize; // 设置总的 workspace 的数值大小，总的 workspace 空间由框架来申请并管理。
    ...
}
```
