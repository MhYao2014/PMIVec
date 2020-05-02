/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "loss.h"
#include "utils.h"

#include <cmath>

namespace fasttext {

    constexpr int64_t SIGMOID_TABLE_SIZE = 512;
    constexpr int64_t MAX_SIGMOID = 8;
    constexpr int64_t LOG_TABLE_SIZE = 512;

    real NegativeSamplingLoss::forward(
            const std::vector<int32_t>& targets,
            int32_t targetIndex,
            Model::State& state,
            real lr,
            bool backprop) {
        assert(targetIndex >= 0);
        assert(targetIndex < targets.size());
        int32_t target = targets[targetIndex];
        real loss = binaryLogistic(target, state, true, lr, backprop);

        for (int32_t n = 0; n < neg_; n++) {
            auto negativeTarget = getNegative(target, state.rng);
            loss += binaryLogistic(negativeTarget, state, false, lr, backprop);
        }
        return loss;
    }

    real BinaryLogisticLoss::binaryLogistic(
            int32_t target,
            Model::State& state,
            bool labelIsPositive,
            real lr,
            bool backprop) const {
        real score = sigmoid(wo_->dotRow(state.hidden, target));
        if (backprop) {
            real alpha = lr * (real(labelIsPositive) - score);
            state.grad.addRow(*wo_, target, alpha);
            wo_->addVectorToRow(state.hidden, target, alpha);
        }
        if (labelIsPositive) {
            return -log(score);
        } else {
            return -log(1.0 - score);
        }
    }

    int32_t NegativeSamplingLoss::getNegative(
            int32_t target,
            std::minstd_rand& rng) {
        int32_t negative;
        do {
            negative = negatives_[uniform_(rng)];
        } while (target == negative);
        return negative;
    }

    NegativeSamplingLoss::NegativeSamplingLoss(
            std::shared_ptr<Matrix>& wo,
            int neg,
            const std::vector<int64_t>& targetCounts)
            : BinaryLogisticLoss(wo), neg_(neg), negatives_(), uniform_() {
        real z = 0.0;
        for (size_t i = 0; i < targetCounts.size(); i++) {
            z += pow(targetCounts[i], 0.5);
        }
        for (size_t i = 0; i < targetCounts.size(); i++) {
            real c = pow(targetCounts[i], 0.5);
            for (size_t j = 0; j < c * NegativeSamplingLoss::NEGATIVE_TABLE_SIZE / z;
                 j++) {
                negatives_.push_back(i);
            }
        }
        uniform_ = std::uniform_int_distribution<size_t>(0, negatives_.size() - 1);
    }

    Loss::Loss(std::shared_ptr<Matrix>& wo) : wo_(wo) {
        t_sigmoid_.reserve(SIGMOID_TABLE_SIZE + 1);
        for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
            real x = real(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
            t_sigmoid_.push_back(1.0 / (1.0 + std::exp(-x)));
        }

        t_log_.reserve(LOG_TABLE_SIZE + 1);
        for (int i = 0; i < LOG_TABLE_SIZE + 1; i++) {
            real x = (real(i) + 1e-5) / LOG_TABLE_SIZE;
            t_log_.push_back(std::log(x));
        }
    }

    real Loss::log(real x) const {
        if (x > 1.0) {
            return 0.0;
        }
        int64_t i = int64_t(x * LOG_TABLE_SIZE);
        return t_log_[i];
    }

    real Loss::sigmoid(real x) const {
        if (x < -MAX_SIGMOID) {
            return 0.0;
        } else if (x > MAX_SIGMOID) {
            return 1.0;
        } else {
            int64_t i =
                    int64_t((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
            return t_sigmoid_[i];
        }
    }

    BinaryLogisticLoss::BinaryLogisticLoss(std::shared_ptr<Matrix>& wo)
            : Loss(wo) {}
} // namespace fasttext
