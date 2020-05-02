/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <time.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <queue>
#include <set>
#include <tuple>

#include "args.h"
#include "densematrix.h"
#include "dictionary.h"
#include "matrix.h"
#include "model.h"
#include "real.h"
#include "utils.h"
#include "vector.h"

namespace fasttext {

class FastText {
 protected:
  std::shared_ptr<Args> args_;
  std::shared_ptr<Dictionary> dict_;
  std::shared_ptr<Matrix> input_;
  std::shared_ptr<Matrix> output_;
  std::shared_ptr<Model> model_;
  std::atomic<int64_t> tokenCount_{};
  std::atomic<real> loss_{};
  std::chrono::steady_clock::time_point start_;
  std::unique_ptr<DenseMatrix> wordVectors_;
  std::exception_ptr trainException_;

  void startThreads();
  void addInputVector(Vector&, int32_t) const;
  void trainThread(int32_t);
  void printInfo(real, real, std::ostream&);
  std::shared_ptr<Matrix> createRandomMatrix() const;
  std::shared_ptr<Matrix> createTrainOutputMatrix() const;
  std::vector<int64_t> getTargetCounts() const;
  std::shared_ptr<Loss> createLoss(std::shared_ptr<Matrix>& output);
  void skipgram(Model::State& state, real lr, const std::vector<int32_t>& line);
  bool keepTraining(const int64_t ntokens) const;
 public:
  FastText();

  void getWordVector(Vector& vec, const std::string& word) const;

  void saveVectors(const std::string& filename);

  void saveOutput(const std::string& filename);

  void train(const Args& args);
};
} // namespace fasttext
