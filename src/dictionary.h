/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <istream>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "args.h"
#include "real.h"

namespace fasttext {

typedef int32_t id_type;
enum class entry_type : int8_t { word = 0, label = 1 };

struct entry {
  std::string word;
  int64_t count;
  int64_t dictId;
};

class Dictionary {
 protected:
  static const int32_t MAX_VOCAB_SIZE = 30000000;
  static const int32_t MAX_LINE_SIZE = 1024;

  int32_t find(const std::string&) const;
  int32_t find(const std::string&, uint32_t h) const;
  void initTableDiscard();
  void reset(std::istream&) const;

  std::shared_ptr<Args> args_;
  std::vector<int32_t> word2int_;
  std::vector<entry> words_;

  std::vector<real> pdiscard_;
  int32_t size_;
  int32_t nwords_;
  int64_t ntokens_;

 public:
  static const std::string EOS;

  explicit Dictionary(std::shared_ptr<Args>);
  int32_t nwords() const;
  int64_t ntokens() const;
  int32_t getId(const std::string&) const;
  bool discard(int32_t, real) const;
  std::string getWord(int32_t) const;
  uint32_t hash(const std::string& str) const;
  void add(const std::string&);
  bool readWord(std::istream&, std::string&) const;
  void readFromFile(std::istream&);
  std::vector<int64_t> getCounts(entry_type) const;
  int32_t getLine(std::istream&, std::vector<int32_t>&, std::minstd_rand&)
      const;
  void threshold(int64_t, int64_t);
};

} // namespace fasttext