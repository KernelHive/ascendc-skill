##### GetInputShape

## 函数功能
获取算子输入张量的实际存储形状。

## 函数原型
```cpp
const gert::StorageShape* GetInputShape(int64_t index) const
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| index | 输入 | 算子输入索引。<br>这里的输入索引是指算子实例化后实际的索引，不是原型定义中的索引。 |

## 返回值说明
返回 StorageShape 结构体，表示该输入张量的存储形状。

## 约束说明
使用时必须确保：不要对返回的 StorageShape 做修改，也不要试图释放它。

## 调用示例
```cpp
gert::StorageShape* GetInputShape(OpCheckContext* context) {
    int64_t index = 9;
    gert::StorageShape* shape = context->GetInputShape(index);
    // ...
}
```
