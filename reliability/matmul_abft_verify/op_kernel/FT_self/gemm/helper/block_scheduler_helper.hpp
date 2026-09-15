/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FTSELF_GEMM_HELPER_BLOCK_SCHEDULER_HELPER_HPP
#define FTSELF_GEMM_HELPER_BLOCK_SCHEDULER_HELPER_HPP

#include <cstdint>

namespace FTSelf::Gemm::helper {

struct MnBlockInfo {
    uint32_t mIndex;
    uint32_t nIndex;
    uint32_t mOffset;
    uint32_t nOffset;
    uint32_t actualM;
    uint32_t actualN;
};

template <uint32_t SwizzleM = 3>
class MnTileScheduler {
public:
    __aicore__ MnTileScheduler(uint32_t problemM, uint32_t problemN, uint32_t tileM, uint32_t tileN)
        : problemM_(problemM), problemN_(problemN), tileM_(tileM), tileN_(tileN),
          mLoops_(tileM == 0 ? 0 : (problemM + tileM - 1) / tileM),
          nLoops_(tileN == 0 ? 0 : (problemN + tileN - 1) / tileN)
    {
    }

    __aicore__ uint32_t GetTaskCount() const
    {
        return mLoops_ * nLoops_;
    }

    __aicore__ uint32_t GetMLoops() const
    {
        return mLoops_;
    }

    __aicore__ uint32_t GetNLoops() const
    {
        return nLoops_;
    }

    __aicore__ MnBlockInfo GetBlockInfo(uint32_t taskIndex) const
    {
        uint32_t innerIndex = taskIndex % GetTaskCount();
        uint32_t tileGroupCount = (mLoops_ + SwizzleM - 1) / SwizzleM;
        uint32_t tileGroupIndex = innerIndex / (SwizzleM * nLoops_);
        uint32_t indexInGroup = innerIndex % (SwizzleM * nLoops_);

        uint32_t rowsInGroup = SwizzleM;
        if (tileGroupIndex == tileGroupCount - 1) {
            rowsInGroup = mLoops_ - SwizzleM * tileGroupIndex;
        }

        uint32_t mIndex = tileGroupIndex * SwizzleM + indexInGroup % rowsInGroup;
        uint32_t nIndex = indexInGroup / rowsInGroup;
        if (tileGroupIndex % 2 == 1) {
            nIndex = nLoops_ - nIndex - 1;
        }

        uint32_t mOffset = mIndex * tileM_;
        uint32_t nOffset = nIndex * tileN_;
        uint32_t actualM = (mIndex == mLoops_ - 1) ? (problemM_ - mOffset) : tileM_;
        uint32_t actualN = (nIndex == nLoops_ - 1) ? (problemN_ - nOffset) : tileN_;
        return {mIndex, nIndex, mOffset, nOffset, actualM, actualN};
    }

private:
    uint32_t problemM_;
    uint32_t problemN_;
    uint32_t tileM_;
    uint32_t tileN_;
    uint32_t mLoops_;
    uint32_t nLoops_;
};

} // namespace FTSelf::Gemm::helper

#endif // FTSELF_GEMM_HELPER_BLOCK_SCHEDULER_HELPER_HPP
