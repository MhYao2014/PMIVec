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

    void NegativeSamplingLoss::forward(
            const std::vector<int32_t>& targets,
            int32_t targetIndex,
            int32_t secTargetIdex,
            Model::State& state,
            real lr,
            bool backprop) {
        assert(targetIndex >= 0);
        assert(targetIndex < targets.size());
        int32_t target = targets[targetIndex], secTarget = targets[secTargetIdex];
        real tmpLossSecond = 0.0, tmpLossFirst = 0.0;
        state.outputVec.zero();state.outputVec.addRow(*wo_,target);
        state.secOutVec.zero();state.secOutVec.addRow(*wo_,secTarget);
        biLogiFirst(target, state, true, lr, backprop, tmpLossFirst);
        biLogiSecond(target, secTarget, state, true, lr, backprop, tmpLossSecond);
        for (int32_t n = 0; n < neg_; n++) {
            auto negativeTarget = getNegative(target, state.rng);
            state.outputVec.zero();state.outputVec.addRow(*wo_,negativeTarget);
            biLogiFirst(negativeTarget, state, false, lr, backprop,tmpLossFirst);
        }
        long long negOutId,secNegOutId;
        for (int secNegI = 0; secNegI < neg_ % 2; secNegI++) { // second term's neg part
            for (int secNegJ = 0; secNegJ < neg_ % 2; secNegJ++) {
                negOutId = getNegative(target,state.rng);
                secNegOutId = getNegative(secTarget,state.rng);
                state.outputVec.zero();state.outputVec.addRow(*wo_,negOutId,1.0);
                state.secOutVec.zero();state.secOutVec.addRow(*wo_,secNegOutId,1.0);
                biLogiSecond(negOutId,secNegOutId,state, false, lr, backprop,tmpLossSecond);
            }
        }
        state.incrementLoss(tmpLossFirst,tmpLossSecond);
    }

    void BinaryLogisticLoss::biLogiFirst(
            int32_t target,
            Model::State& state,
            bool labelIsPositive,
            real lr,
            bool backprop,
            real& tmpLossFirst) const {
        real inner = state.inputVec.dotMul(state.outputVec,1.0);
        real score = sigmoid(inner);
        if (backprop) {
            real alpha = (real(labelIsPositive) - score);
//            state.outputVec.addVector(state.inputVec,-1 * inner);
            state.inputGrad.addVector(state.outputVec, -1 * alpha);
            wo_->addVectorToRow(state.inputVec, target, lr * alpha);
        }
        if (labelIsPositive) {
            tmpLossFirst += -log(score);
        } else {
            tmpLossFirst += -log(1.0 - score);
        }
    }

    void BinaryLogisticLoss::biLogiSecond(int32_t outId,int32_t secOutId,Model::State& state,bool labelIsPositive,
            real lr,bool backprop,real &tmpLossSecond) const {
        state.sumOutVec.zero();
        state.sumOutVec.addVector(state.outputVec,1.0);
        state.sumOutVec.addVector(state.secOutVec,1.0);
        real inner = state.inputVec.dotMul(state.sumOutVec,1.0);
        real score = sigmoid(inner);
        if (backprop) {
            real alpha = (real(labelIsPositive) - score);
//            state.sumOutVec.addVector(state.inputVec, -1 * inner);
            state.inputGrad.addVector(state.sumOutVec, -1 * alpha);
            wo_->addVectorToRow(state.inputVec, outId, lr * alpha);
            wo_->addVectorToRow(state.inputVec, secOutId, lr * alpha);
        }
        if (labelIsPositive) {
            tmpLossSecond += -log(score);
        } else {
            tmpLossSecond += -log(1.0 - score);
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
} // namespace fasttext
