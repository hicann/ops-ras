/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_FTSELF_GEMV_BLOCK_BLOCK_GEMV_HPP
#define FTSELF_FTSELF_GEMV_BLOCK_BLOCK_GEMV_HPP



#include "../../gemv/helper.hpp"
#include "../../gemv/tile/gemv_tile_copy.hpp"
#include "../../gemv/tile/tile_fault_compare.hpp"
#include "../../gemv/tile/tile_fault_sum.hpp"
#include "../../gemv/tile/tile_matmul_elem_add.hpp"

namespace FTSelf::Gemv::Block {


/*
struct BlockSumMaxNoSplitK <
    FTSelf::Gemv::GemvAtlasA2,
    FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR,
    UBTileShape_,
    AType_,
    XType_,
    YType_,
    BiasType_,
    TileCopy_,
    TileFaultSum_
>
*/
template <
    class DispatchPolicy,
    FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM ALGO_TYPE,
    class... Args
>
struct BlockSumMaxNoSplitK {
    static_assert(FTSelf::helper::DEPENDENT_FALSE<DispatchPolicy>, "BlockSumMaxNoSplitK is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM ALGO_TYPE,
    class... Args
>
struct BlockSumMaxNoSplitKBF {
    static_assert(FTSelf::helper::DEPENDENT_FALSE<DispatchPolicy>, "BlockSumMaxNoSplitKBF is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM ALGO_TYPE_,
    FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE_,
    class... Args
>
struct BlockFTSumNoSplitK {
    static_assert(FTSelf::helper::DEPENDENT_FALSE<DispatchPolicy>, "BlockFTSumNoSplitK is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM ALGO_TYPE_,
    FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE_,
    FTSelf::Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    FTSelf::Gemv::helper::FT_COMP_TYPE COMP_TYPE_,
    class... Args
>
struct BlockFTGemvNoSplitK {
    static_assert(FTSelf::helper::DEPENDENT_FALSE<DispatchPolicy>, "BlockFTGemvNoSplitK is not implemented for this DispatchPolicy");
};

#define FTSELF_DECLARE_CE_BLOCK_PRIMARY(BlockName)                                                   \
    template <                                                                                       \
        class DispatchPolicy,                                                                        \
        FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM ALGO_TYPE_,                                    \
        FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE_,                                     \
        FTSelf::Gemv::helper::FT_ENC_TYPE ENC_TYPE_,                                                 \
        FTSelf::Gemv::helper::FT_COMP_TYPE COMP_TYPE_,                                               \
        FTSelf::Gemv::helper::FT_ABE_TYPE ABE_TYPE_,                                                 \
        class... Args>                                                                                \
    struct BlockName {                                                                               \
        static_assert(FTSelf::helper::DEPENDENT_FALSE<DispatchPolicy>,                               \
            #BlockName " is not implemented for this DispatchPolicy");                              \
    }

FTSELF_DECLARE_CE_BLOCK_PRIMARY(BlockFTGemvCENoSplitK);
FTSELF_DECLARE_CE_BLOCK_PRIMARY(BlockFTGemvCENoSplitKPreload);

#undef FTSELF_DECLARE_CE_BLOCK_PRIMARY

template <
    class DispatchPolicy,
    class... Args
>
struct BlockSumGemv {
    static_assert(FTSelf::helper::DEPENDENT_FALSE<DispatchPolicy>, "BlockSumGemv is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    FTSelf::Gemv::helper::FT_COMP_TYPE COMP_TYPE_,
    class... Args
>
struct BlockCompare {
    static_assert(FTSelf::helper::DEPENDENT_FALSE<DispatchPolicy>, "BlockCompare is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    FTSelf::Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    class... Args
>
struct BlockThresholdCalc {
    static_assert(FTSelf::helper::DEPENDENT_FALSE<DispatchPolicy>, "BlockThresholdCalc is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM ALGO_TYPE_,
    FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE_,
    FTSelf::Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    FTSelf::Gemv::helper::FT_COMP_TYPE COMP_TYPE_,
    class... Args
>
struct BlockThresholdCalcFused {
    static_assert(FTSelf::helper::DEPENDENT_FALSE<DispatchPolicy>, "BlockThresholdCalc is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    class... Args
>
struct BlockSliceSum {
    static_assert(FTSelf::helper::DEPENDENT_FALSE<DispatchPolicy>, "BlockSliceSum is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    FTSelf::Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    class... Args
>
struct BlockFTGemv {
    static_assert(FTSelf::helper::DEPENDENT_FALSE<DispatchPolicy>, "BlockFTGemv is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    FTSelf::Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    class... Args
>
struct BlockFTGemvDouble {
    static_assert(FTSelf::helper::DEPENDENT_FALSE<DispatchPolicy>, "BlockFTGemv is not implemented for this DispatchPolicy");
};


template <
    class DispatchPolicy,
    FTSelf::Gemv::helper::FT_AIC_BE_SCHEME BE_SCHEME_,
    class... Args
>
struct BlockFTGemvBe {
    static_assert(FTSelf::helper::DEPENDENT_FALSE<DispatchPolicy>, "BlockFTGemv is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    class... Args
>
struct BlockSumGemvPingPong {
    static_assert(FTSelf::helper::DEPENDENT_FALSE<DispatchPolicy>, "BlockSumGemvPingPong is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    FTSelf::Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    bool  VECTORIZED_TRANS_,
    class... Args
>
struct BlockFTGemvVectorized {
    static_assert(FTSelf::helper::DEPENDENT_FALSE<DispatchPolicy>, "BlockFTGemvVectorized is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    class... Args
>
struct BlockMatrixAdd{
    static_assert(FTSelf::helper::DEPENDENT_FALSE<DispatchPolicy>, "BlockMatrixAdd is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    class... Args
>
struct BlockMatrixTranspose{
    static_assert(FTSelf::helper::DEPENDENT_FALSE<DispatchPolicy>, "BlockMatrixTranspose is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    class... Args
>
struct BlockMatrixAddVectorized{
    static_assert(FTSelf::helper::DEPENDENT_FALSE<DispatchPolicy>, "BlockMatrixAddVectorized is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE_,
    class... Args
>
struct BlockSliceKMNSum {
    static_assert(FTSelf::helper::DEPENDENT_FALSE<DispatchPolicy>, "BlockSliceKMNSum is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    FTSelf::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE_,
    class... Args
>
struct BlockSliceKMNSumVectorized {
    static_assert(FTSelf::helper::DEPENDENT_FALSE<DispatchPolicy>, "BlockSliceKMNSumVectorized is not implemented for this DispatchPolicy");
};

}  // namespace FTSelf::Gemv::Block

#include "../../gemv/block/block_sum_gemv_aiv.hpp"
#include "../../gemv/block/block_gemv_aic_FT_BE_row_complete.hpp"
#include "../../gemv/block/block_b_a_red_no_splitk_robust.hpp"
#include "../../gemv/block/block_b_a_red_no_splitk_robust_bf.hpp"
#include "../../gemv/block/block_slicekmn_reduce_sum.hpp"
#include "../../gemv/block/block_slicekmn_reduce_sum_fused_robust.hpp"
#include "../../gemv/block/block_gemv_aic_FT_BE_row_complete_bf.hpp"
#include "../../gemv/block/block_sum_aiv_ceft_no_splitk_thre_robust_preload.hpp"


#endif // FTSELF_FTSELF_GEMV_BLOCK_BLOCK_GEMV_HPP
