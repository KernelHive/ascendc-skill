#### ICPU_SET_TILING_KEY

## 功能说明

用于指定本次CPU调测使用的tilingKey。调测执行时，将只执行算子核函数中该tilingKey对应的分支。

## 函数原型

```c
ICPU_SET_TILING_KEY(tilingKey)
```

## 参数说明

| 参数名   | 输入/输出 | 描述                                                         |
|----------|-----------|--------------------------------------------------------------|
| tilingKey | 输入      | 指定本次CPU调测使用的tilingKey，参数类型为`int32_t`。        |

## 返回值说明

无

## 约束说明

- 未使用该接口设置tilingKey的情况下，tilingKey将会为默认值0，在调测执行时，会有告警提示Tiling Key是0，并继续进行调测。如果核函数中有tilingKey分支，将会执行tilingKey为0的分支，其他tilingKey对应的分支不会执行。
- tilingKey建议传入正整数，如果设置为负数或者0，将会告警并继续调测。如果传入0，将会执行tilingKey为0的分支；tilingKey传入负数，将导致未定义的行为。
- 该接口需要在`ICPU_RUN_KF`前调用。

## 调用示例

```c
ICPU_SET_TILING_KEY(10086);
ICPU_RUN_KF(sort_kernel0, coreNum, (uint8_t*)x, (uint8_t*)y);
```
