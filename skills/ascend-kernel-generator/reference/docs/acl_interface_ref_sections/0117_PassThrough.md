### PassThrough

该算子对应Caffe框架下的Reorg算子。定制方法请参见《ATC离线模型编译工具用户指南》> 专题 > 定制网络修改（Caffe）> 扩展算子规则。

## 输入

### x

- **是否必填**: 必填
- **数据类型**: float16、float32、int8、uint8、int16、uint16、int32、uint32、int64、uint64
- **参数解释**: 将通道数据转移到平面上，或反过来操作，如下为 reverse==True，输入Shape为[batch, C*stride*stride, H, W]，输出Shape为[batch,C,H * stride, W *stride]
- **规格限制**: 无

## 属性

### stride

- **是否必填**: 非必填
- **数据类型**: int
- **参数解释**: 平面或通道数扩大的倍数，默认2
- **规格限制**: float16

### reverse

- **是否必填**: 非必填
- **数据类型**: bool
- **参数解释**: 为True时相当于Depthtospace，为False时相当于spacetoDepth
- **规格限制**: float16

## 输出

### y

- **是否必填**: 必填
- **数据类型**: float16、float32、int8、uint8、int16、uint16、int32、uint32、int64、uint64
- **参数解释**: 将通道数据转移到平面上，或反过来操作，如下为reverse==True，输入Shape为[batch, C*stride*stride, H, W]，输出Shape为[batch,C,H * stride, W *stride]
- **规格限制**: 无
