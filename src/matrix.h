//
// Created by mhyao on 20-1-21.
//
#ifndef ASSVEC_MATRIX_H
#define ASSVEC_MATRIX_H
#pragma once
#include <vector>
#include <cstdint>
// 因为Matrix和Vector相互引用,这里是前置申明
class Vector;

class Matrix {
protected:
    int64_t row_;
    int64_t col_;
    std::vector<double> data_;

public:
    // 构造函数
    Matrix();
    explicit Matrix(int64_t, int64_t);
    // 析构函数
    virtual ~Matrix() noexcept = default;

    // 常用方法合集
    // 返回指向vector内存首地址的指针
    inline double * data() {
        return data_.data();
    }
    // at算符
    inline double & at(int64_t i, int64_t j) {
        return data_[i * col_ + j];
    }
    // 读取行数
    inline int64_t rows() {
        return row_;
    }
    // 读取列数
    inline int64_t cols() {
        return col_;
    }

    int64_t size();
    // 元素清零
    void zero();
    // 均匀随机初始化
    void uniform(double);
    // 计算该向量的标量乘法
    void scalerMul(double a);
    // 计算第i行的摸长
    double l2NormRow(int64_t i);
    // 只对第i行乘以一个标量
    void scalerMulRow(double a, int64_t i);
    // 计算一个和该向量第i行的内积
    double dotRow(Vector& source, int64_t, double);
    // 计算将外部向量加到某一行上,实现参数更新的效果
    void addVectorToRow(Vector& vec, int64_t i, double a);
    // 计算将某一行加到外部向量上,实现取出某一行的效果
    void addRowToVector(Vector& vec, int64_t i, double a);
    // 把矩阵的某部分加到另一个向量中
    void addPart(Vector& vec, int64_t Id, int64_t subId, double a);
    // 计算矩阵间的加法
    void addMatrix(Matrix& mat, double a);
    // 计算与一个拉直矩阵的加法
    void addFlatMatrix(Matrix& mat, double a, int Id);
    // 保存矩阵
    void saveMat2Row(int Id, Matrix &mat);
    // 把一个矩阵加到自己的某一行中
    void addMat2Row(Matrix &mat, int Id, double a);
    // 把矩阵的某一行加到另一个矩阵的另一行
    void addRow2Row(Matrix &mat, int selfId, int Id, double a);
    // 把矩阵某一行换为某向量
    void replaceRowWithVector(Vector& vec, int64_t i, double a);
};

#endif