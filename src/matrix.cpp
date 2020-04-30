//
// Created by mhyao on 20-1-21.
//
#include "matrix.h"
#include "vector.h"
#include <vector>
#include <stdio.h>
#include <random>
#include <assert.h>
#include <stdexcept>

Matrix::Matrix(): row_(0), col_(0) {}

Matrix::Matrix(int64_t row, int64_t col): row_(row), col_(col) ,data_(row * col){}

void Matrix::zero() {
    std::fill(data_.begin(), data_.end(), 0.0);
}

void Matrix::uniform(double a) {
    std::minstd_rand rng(time(NULL));
    std::uniform_real_distribution<> uniform(-a, a);
    for (int64_t i=0; i < (row_ * col_); i++) {
        data_[i] = uniform(rng);
    }
}

void Matrix::scalerMul(double a) {
    for (int64_t j = 0; j < data_.size(); j++) {
        data_[j] = a * data_[j];
    }
}

void Matrix::scalerMulRow(double a, int64_t i) {
    for (auto j = 0; j < col_; j++) {
        data_[i * col_ + j] *= a;
    }
}

double Matrix::l2NormRow(int64_t i) {
    auto norm = 0.0;
    for (auto j = 0; j < col_; j++) {
        norm += at(i, j) * at(i, j);
    }
    if (std::isnan(norm)) {
        throw std::runtime_error("Encountered NaN.");
    }
    return std::sqrt(norm);
}

double Matrix::dotRow(Vector& vec, int64_t i, double a) {
    assert(i >= 0);
    assert(i < row_);
    assert(vec.size() == col_);
    double d = 0.0;
    for (int64_t j = 0; j < col_; j++) {
        d += at(i,j) * vec[j];
    }
    if (std::isnan(d)) {
        throw std::runtime_error("Encountered NaN.");
    }
    return d;
}

void Matrix::addVectorToRow(Vector &vec, int64_t i, double a) {
    assert(i >= 0);
    assert(i < row_);
    assert(vec.size() == col_);
    for (int64_t j = 0; j < col_; j++) {
        data_[i * col_ + j] += a * vec[j];
    }
}

void Matrix::replaceRowWithVector(Vector &vec, int64_t i, double a) {
    assert(i >= 0);
    assert(i < row_);
    assert(vec.size() == col_);
    for (int64_t j = 0; j < col_; j++) {
        data_[i * col_ + j] = a * vec[j];
    }
}

void Matrix::addRowToVector(Vector &vec, int64_t i, double a) {
    assert(i >= 0);
    assert(i < row_);
    assert(vec.size() == col_);
    for (int64_t j = 0; j < col_; j++) {
        vec[j] += a * at(i, j);
    }
}

void Matrix::addPart(Vector &vec, int64_t Id, int64_t subId, double a) {
    assert(Id >= 0);
    assert(Id < row_);
    assert(vec.size() + subId <= col_);
    for (int64_t j = 0; j < vec.size(); j++) {
        vec[j] += a * data_[Id * row_ + subId + j];
    }
}

void Matrix::addMatrix(Matrix &mat, double a) {
    assert(row_ == mat.row_);
    assert(col_ == mat.col_);
    for (int64_t j = 0; j < mat.data_.size(); j++) {
        data_[j] += a * mat.data_[j];
    }
}

void Matrix::addFlatMatrix(Matrix &mat, double a, int rowId){
    // 这里没有检查是否会越界。传入的mat其实是一个张量。
    for (int64_t j = 0; j < mat.col_; j++) {
        data_[j] += a * mat.data_[rowId * mat.col_ + j];
    }
}

void Matrix::saveMat2Row(int rowId, Matrix &mat) {
    for (int64_t j = 0; j < mat.data_.size(); j++) {
        data_[col_ * rowId + j] = mat.data_[j];
    }
}

void Matrix::addMat2Row(Matrix &mat, int rowId, double a) {
    for (int64_t j = 0; j < mat.data_.size(); j++) {
        data_[col_ * rowId + j] += a * mat.data_[j];
    }
}

void Matrix::addRow2Row(Matrix &mat, int selfRowId, int rowId, double a) {
    assert(col_ == mat.cols());
    for (int64_t j = 0; j < mat.cols(); j++) {
        data_[col_ * selfRowId + j] += a * mat.data_[mat.cols() * rowId + j];
    }
}

int64_t Matrix::size() {
    return data_.size();
}