##### GetWorkspaceBytes

## 函数功能

获取 Workspace 大小。

## 函数原型

```cpp
std::vector<int64_t> GetWorkspaceBytes() const
```

## 参数说明

无

## 返回值说明

返回 Workspace 大小配置信息。

## 约束说明

无

## 调用示例

```cpp
std::vector<int64_t> GetWorkspaceBytes(ExeResGenerationContext* context) {
  std::vector<int64_t> workspaceList = context->GetWorkspaceBytes();
  // ...
}
```
