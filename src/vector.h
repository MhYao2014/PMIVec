//
// Created by mhyao on 20-1-21.
//

#ifndef ASSVEC_VECTOR_H
#define ASSVEC_VECTOR_H

#include <cstdint>
#include <ostream>
#include <vector>
#include "matrix.h"

// 因为Matrix和Vector相互引用,这里是前置申明

class Vector {
protected:
    std::vector<double> data_;

public:
    // 构造函数
    explicit Vector(int64_t);
    Vector(Vector &&) noexcept = default;
    Vector&operator=(Vector&&) = default;

    // 暴露数据
    inline double * data() {
        return data_.data();
    }
    // 返回向量长度
    inline int64_t size() {
        return data_.size();
    }
    // 设置下标取值操作
    inline double &operator[](int64_t i) {
        return data_[i];
    }
    // 元素清零
    void zero();
    // 计算模长
    double norm();
    // 计算该向量的标量乘法
    void scalerMul(double a);
    // 向量间计算内积
    double dotMul(Vector & vec, double a);
    // 计算将外部向量加到该向量上
    void addVector(Vector& vec, double a);
    // 计算将矩阵某一行加到外部向量上,实现取出某一行的效果
    void addRow(Matrix& mat, int64_t i, double a);
    // 计算将张量中的某个向量加到外部向量上
    void addRowTensor(Matrix& mat, int64_t Id, int64_t subId, double a);
    // 随机初始化该向量
    void uniform(double bound);
};

#endif //ASSVEC_VECTOR_H