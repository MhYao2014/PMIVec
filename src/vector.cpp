//
// Created by mhyao on 20-1-22.
//
#include "vector.h"
#include <cmath>
#include <assert.h>
#include <random>

Vector::Vector(int64_t m): data_(m) {}

void Vector::zero() {
    std::fill(data_.begin(), data_.end(), 0.0);
}

double Vector::norm() {
    double sum = 0;
    for (int64_t i = 0; i < size(); i++) {
        sum += data_[i] * data_[i];
    }
    return std::sqrt(sum);
}

void Vector::scalerMul(double a) {
    for (int64_t i = 0; i < size(); i++) {
        data_[i] *= a;
    }
}

double Vector::dotMul(Vector &vec, double a) {
    assert(vec.size() == size());
    double result = 0;
    for (int64_t i = 0; i < size(); i++) {
        result += data_[i] * vec.data_[i] * a;
    }
    return result;
}

void Vector::addVector(Vector &vec, double a) {
    assert(size() == vec.size());
    for (int64_t i = 0; i < size(); i++) {
        data_[i] += a * vec.data_[i];
    }
}

void Vector::addRow(Matrix &mat, int64_t i, double a) {
    assert(i >= 0);
    assert(i < mat.rows());
    assert(size() == mat.cols());
    mat.addRowToVector(*this, i, a);
}

void Vector::addRowTensor(Matrix &mat, int64_t Id, int64_t subId, double a) {
    assert(Id >= 0);
    assert(Id < mat.rows());
    assert(size() + subId <= mat.cols());
    mat.addPart(*this, Id, subId, a);
}

void Vector::uniform(double bound) {
    std::minstd_rand rng(time(NULL));
    std::uniform_real_distribution<> uniform(-bound, bound);
    for (int i = 0; i < data_.size(); i++) {
        data_[i] = uniform(rng);
    }
}