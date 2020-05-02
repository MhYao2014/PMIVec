/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <random>
#include <vector>

#include "matrix.h"
#include "model.h"
#include "real.h"
#include "utils.h"
#include "vector.h"

namespace fasttext {

    class Loss {
    private:

    protected:
        std::vector<real> t_sigmoid_;
        std::vector<real> t_log_;
        std::shared_ptr<Matrix>& wo_;

        real log(real x) const;
        real sigmoid(real x) const;

    public:
        explicit Loss(std::shared_ptr<Matrix>& wo);
        virtual ~Loss() = default;

        virtual void forward(
                const std::vector<int32_t>& targets,
                int32_t targetIndex,
                int32_t secTargetIdex,
                Model::State& state,
                real lr,
                bool backprop) = 0;
    };

    class BinaryLogisticLoss : public Loss {
    protected:
        void biLogiFirst(
                int32_t target,
                Model::State& state,
                bool labelIsPositive,
                real lr,
                bool backprop,
                real &tmpLossFirst) const;

        void biLogiSecond(
                int32_t outId,
                int32_t secOutId,
                Model::State& state,
                bool labelIsPositive,
                real lr,
                bool backprop,
                real &tmpLossSecond) const;

    public:
        explicit BinaryLogisticLoss(std::shared_ptr<Matrix>& wo);
        virtual ~BinaryLogisticLoss() noexcept override = default;
    };

    class NegativeSamplingLoss : public BinaryLogisticLoss {
    protected:
        static const int32_t NEGATIVE_TABLE_SIZE = 10000000;

        int neg_;
        std::vector<int32_t> negatives_;
        std::uniform_int_distribution<size_t> uniform_;
        int32_t getNegative(int32_t target, std::minstd_rand& rng);

    public:
        explicit NegativeSamplingLoss(
                std::shared_ptr<Matrix>& wo,
                int neg,
                const std::vector<int64_t>& targetCounts);
        ~NegativeSamplingLoss() noexcept override = default;

        void forward(
                const std::vector<int32_t>& targets,
                int32_t targetIndex,
                int32_t secTargetIdex,
                Model::State& state,
                real lr,
                bool backprop) override;
    };

} // namespace fasttext
