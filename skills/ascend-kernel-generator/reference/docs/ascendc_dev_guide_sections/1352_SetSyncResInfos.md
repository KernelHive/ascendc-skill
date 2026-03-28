##### SetSyncResInfos

## 函数功能

将传入的同步资源信息列表（sync_info_vec）设置到当前对象中，用于配置图执行时的同步资源行为。

## 函数原型

```cpp
ge::graphStatus SetSyncResInfos(std::vector<SyncResInfo> &sync_info_vec) const
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| sync_info_vec | 输入 | 同步资源信息列表。<br>std::vector<SyncResInfo>类型。 |

## 返回值说明

返回 graphStatus 状态码。

## 约束说明

无

## 调用示例

```cpp
ge::graphStatus SetSyncResInfos(ExeResGenerationContext* context) {
    std::vector<SyncResInfo> sync_info_vec;
    ge::graphStatus status = context->SetSyncResInfos(sync_info_vec);
    // ...
}
```
