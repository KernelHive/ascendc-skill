###### Comment

## 功能说明
设置算子属性的注释。

## 函数原型
```cpp
OpAttrDef &Comment(const char *comment)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| comment | 输入 | 注释内容。 |

## 返回值说明
算子属性定义，OpAttrDef请参考 OpAttrDef。

## 约束说明
无

## 调用示例
```cpp
this->Attr("attrname").Comment("Attr cmt 1");
```
