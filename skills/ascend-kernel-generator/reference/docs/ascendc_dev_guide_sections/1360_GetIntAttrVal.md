##### GetIntAttrVal

## 函数功能
获取算子整数类型的属性值。

## 函数原型
```cpp
bool GetIntAttrVal(const char *attr_name, int64_t &val) const
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| attr_name | 输入 | 属性名 |
| val | 输出 | 接收获取到的整数值 |

## 返回值说明
表示是否成功获取到该属性。

## 约束说明
无

## 调用示例
```cpp
bool GetIntAttrVal(ExeResGenerationContext* context) {
    std::string attr_name = "";
    int64_t val;
    auto result = context->GetStrAttrVal(attr_name.c_str(), val);
    // ...
}
```
