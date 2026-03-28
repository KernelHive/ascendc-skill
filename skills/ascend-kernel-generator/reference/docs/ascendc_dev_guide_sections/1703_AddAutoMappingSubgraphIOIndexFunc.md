##### AddAutoMappingSubgraphIOIndexFunc

## 函数功能
注册指定网络类型的自动映射函数。

## 函数原型
```cpp
void AddAutoMappingSubgraphIOIndexFunc(domi::FrameworkType framework,
                                       AutoMappingSubgraphIOIndexFunc fun)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| framework | 输入 | 网络类型，FrameworkType类型定义请参考15.2.3.25.4 FrameworkType |
| fun | 输入 | 自动映射输入输出函数，函数类型详见AutoMappingSubgraphIndex |

## 返回值
无

## 异常处理
无

## 约束说明
无
