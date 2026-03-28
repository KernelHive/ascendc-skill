/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file masked_select_v3.cc
 * \brief
 */
#include "masked_select_v3.h"
#include <iostream>
#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "tiling/tiling_api.h"

// tools api
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
namespace ops {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }
}  // namespace ops
namespace optiling {
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)
}  // namespace optiling

namespace
{
    constexpr static uint32_t BLOCK_SIZE = 256;
    constexpr static uint32_t DOUBLE_BUFFER = 2;
    constexpr static float UB_USAGE = 0.65f;
    constexpr uint32_t INT64_LENGTH_IN_INT32 = 2; // INT64 相当于 2个int32长
}

namespace optiling
{
    class MaskedSelectV3Tiling
    {
    public:
        explicit MaskedSelectV3Tiling(gert::TilingContext *context) : tilingContext(context) {};
        ge::graphStatus Init();
        ge::graphStatus RunKernelTiling();
        void TilingDataPrint();

    private:
        
        MaskedSelectV3TilingData tiling;

        gert::TilingContext *tilingContext = nullptr;
        // 总元素个数
        uint64_t totalLength = tilingContext->GetInputShape(0)->GetStorageShape().GetShapeSize();
        
        // ub对齐后长度
        uint64_t totalLengthAlignedWithBlock = 0;

        uint64_t tilingKey = 1;
        uint64_t formerNum = 0;
        uint64_t formerLength = 0;
        uint64_t formerTileNum = 0;
        uint64_t formerTileLength = 0;
        uint64_t formerLastTileLength = 0;

        uint64_t tailNum = 0;
        uint64_t tailLength = 0;
        uint64_t tailTileNum = 0;
        uint64_t tailTileLength = 0;
        uint64_t tailLastTileLength = 0;

        uint64_t blockDim = 0;
        
        // 求单个元素大小
        uint64_t sizeOfDataType = 1;
        uint64_t dataType = tilingContext->GetInputDesc(0)->GetDataType();
    };
    ge::graphStatus MaskedSelectV3Tiling::Init()
    {
        OP_LOGD(tilingContext->GetNodeName(), "Tiling init start.");
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
        uint64_t aivNum = ascendcPlatform.GetCoreNumAiv(); // Vector核数量
        uint64_t ubSize; // ubSize大小
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        switch (dataType)
        {
            case ge::DT_FLOAT:
            case ge::DT_INT32:
            case ge::DT_UINT32:
                sizeOfDataType = sizeof(int32_t);
                break;
            case ge::DT_DOUBLE:
            case ge::DT_INT64:
            case ge::DT_UINT64:
                sizeOfDataType = sizeof(int64_t);
                break;
            case ge::DT_FLOAT16:
            case ge::DT_BF16:
            case ge::DT_INT16:
            case ge::DT_UINT16:
                sizeOfDataType = sizeof(int16_t);
                break;
            case ge::DT_BOOL:
            case ge::DT_INT8:
            case ge::DT_UINT8:
                sizeOfDataType = sizeof(int8_t);
                break;
        }

        // 一个block存放的元素
        uint32_t ALIGN_NUM = BLOCK_SIZE / sizeOfDataType;       // 256/<8>=32

        // ub对齐后长度
        totalLengthAlignedWithBlock = ((totalLength + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;

        float total_fact = sizeof(uint8_t) + sizeOfDataType * 3;
        if (sizeOfDataType == sizeof(int64_t)) {
            total_fact += sizeof(float) + (1.0 / sizeof(int64_t)) * INT64_LENGTH_IN_INT32;
        } else {
            total_fact += sizeof(int16_t) + (1.0 / sizeof(int64_t));
        }
        if (sizeOfDataType == 1){
            total_fact += sizeof(int32_t);
        }
        
        // 核内拆分，策略是尽可能的填满ub_size,最后一包单独处理，
        float tmp = (sizeOfDataType) * 1.0 / total_fact;
        // ub_block在ascend C中不能全部被用来作为输入输出，给了13/20系数。计算出x 一次最多能处理多少block数量
        uint32_t ubBlockNum = static_cast<uint32_t>((ubSize * UB_USAGE * tmp) / BLOCK_SIZE); 
        OP_LOGD(tilingContext->GetNodeName(), "ubBlockNum: %u.", ubBlockNum);
        if (ubBlockNum % DOUBLE_BUFFER != 0)
        {
            ubBlockNum = ubBlockNum - 1;
        }

        OP_LOGD(tilingContext->GetNodeName(), "totalLength: %lu.", totalLength);
        // ub能放的元素个数
        uint32_t ubLength = ubBlockNum * ALIGN_NUM;
        // block数量
        uint32_t ubNum = (totalLengthAlignedWithBlock + ubLength - 1) / ubLength;

        // 运行核数
        blockDim = (ubNum > aivNum) ? aivNum : ubNum;
        tilingContext->SetBlockDim(blockDim);

        tilingKey = sizeOfDataType;
        tilingContext->SetTilingKey(tilingKey);

        // 切分流程
        formerNum = totalLength % blockDim;
        if (formerNum == 0){
            formerNum = blockDim;
        }
        tailNum = blockDim - formerNum;

        formerLength = (totalLength + blockDim -1) / blockDim;
        formerTileNum = (formerLength + ubLength - 1) / ubLength;
        formerTileLength = ubLength;
        formerLastTileLength = formerLength % ubLength;
        if (formerLastTileLength == 0) {
            formerLastTileLength = ubLength;
        }

        if (tailNum > 0) {
            tailLength = (totalLength -formerLength * formerNum) / tailNum; // 一定可能整出
            tailTileNum = (tailLength + ubLength - 1) / ubLength;
            tailTileLength = ubLength;
            tailLastTileLength = tailLength % ubLength;
            if (tailLastTileLength == 0) {
                tailLastTileLength = ubLength;
            }
        }
        
        OP_LOGD(tilingContext->GetNodeName(), "Tiling inited.");
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus MaskedSelectV3Tiling::RunKernelTiling()
    {
        OP_LOGD(tilingContext->GetNodeName(), "Tiling start.");
        tiling.set_formerNum(formerNum);
        tiling.set_formerLength(formerLength);
        tiling.set_formertileNum(formerTileNum);
        tiling.set_formertileLength(formerTileLength);
        tiling.set_formerlasttileLength(formerLastTileLength);

        tiling.set_tailNum(tailNum);
        tiling.set_tailLength(tailLength);
        tiling.set_tailtileNum(tailTileNum);
        tiling.set_tailtileLength(tailTileLength);
        tiling.set_taillasttileLength(tailLastTileLength);
        tiling.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
        tilingContext->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
        uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
        size_t *currentWorkspace = tilingContext->GetWorkspaceSizes(1); // 通过框架获取workspace的指针，GetWorkspaces入参所需workspace的块数。当前限制使用一块。
        size_t usrSize = totalLengthAlignedWithBlock * sizeOfDataType + blockDim * 64;
        OP_LOGD(tilingContext->GetNodeName(), "usrWorkspaceSize: %lu.", usrSize);
        currentWorkspace[0] = usrSize + sysWorkspaceSize; // 设置总的workspace的数值大小，总的workspace空间框架来申请并管理。
        TilingDataPrint();
        OP_LOGD(tilingContext->GetNodeName(), "Tiling end.");
        return ge::GRAPH_SUCCESS;
    }

    void MaskedSelectV3Tiling::TilingDataPrint()
    {
        OP_LOGD(tilingContext->GetNodeName(), "sizeOfDataType: %lu.", sizeOfDataType);
        OP_LOGD(tilingContext->GetNodeName(), "formerNum: %lu.", tiling.get_formerNum());
        OP_LOGD(tilingContext->GetNodeName(), "formerLength: %lu.", tiling.get_formerLength());
        OP_LOGD(tilingContext->GetNodeName(), "formerTileNum: %lu.", tiling.get_formertileNum());
        OP_LOGD(tilingContext->GetNodeName(), "formerTileLength: %lu.", tiling.get_formertileLength());
        OP_LOGD(tilingContext->GetNodeName(), "formerLastTileLength: %lu.", tiling.get_formerlasttileLength());
        OP_LOGD(tilingContext->GetNodeName(), "tailNum: %lu.", tiling.get_tailNum());
        OP_LOGD(tilingContext->GetNodeName(), "tailLength: %lu.", tiling.get_tailLength());
        OP_LOGD(tilingContext->GetNodeName(), "tailTileNum: %lu.", tiling.get_tailtileNum());
        OP_LOGD(tilingContext->GetNodeName(), "tailTileLength: %lu.", tiling.get_tailtileLength());
        OP_LOGD(tilingContext->GetNodeName(), "tailLastTileLength: %lu.", tiling.get_taillasttileLength());
    }

    ge::graphStatus TilingForMaskedSelectV3(gert::TilingContext *context)
    {
        MaskedSelectV3Tiling tilingObject(context);
        if (tilingObject.Init() != ge::GRAPH_SUCCESS)
        {
            VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Init tiling object return failed.");
            return ge::GRAPH_FAILED;
        }
        return tilingObject.RunKernelTiling();
    }

    IMPL_OP_OPTILING(MaskedSelectV3)
        .Tiling(TilingForMaskedSelectV3);
}


