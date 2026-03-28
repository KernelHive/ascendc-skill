##### SetWorkspaceBytes

## 函数功能
设置 Workspace 大小。

## 函数原型
```cpp
void SetWorkspaceBytes(const std::vector<int64_t> &workspace_bytes) const
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| workspace_bytes | 输入 | Workspace 大小配置信息 |

## 返回值说明
无

## 约束说明
无

## 调用示例
```cpp
void SetWorkspaceBytes(ExeResGenerationContext* context) {
  std::vector<int64_t> workspace_bytes;
  context->SetWorkspaceBytes(workspace_bytes);
  // ...
}
```
