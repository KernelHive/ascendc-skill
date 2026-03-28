##### GetSyncResInfos

## 函数功能

返回当前对象中保存的全部同步资源信息（SyncResInfo列表），用于查询或外部使用。

## 函数原型

```cpp
std::vector<SyncResInfo> GetSyncResInfos() const
```

## 参数说明

无

## 返回值说明

返回当前对象中保存的全部同步资源信息（SyncResInfo列表）。

## 约束说明

无

## 调用示例

```cpp
std::vector<SyncResInfo> GetSyncResInfos(ExeResGenerationContext* context) {
    std::vector<SyncResInfo> syncResInfoList = context->GetSyncResInfos();
    // ...
}
```
