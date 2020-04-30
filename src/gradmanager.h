//
// Created by mhyao on 2020/3/25.
//


#ifndef ASSVEC_ADMM_MPI_GRADMANAGER_H
#define ASSVEC_ADMM_MPI_GRADMANAGER_H

#include "vector.h"
#include <random>

class GradManager {
public:
    std::minstd_rand rng;
    long long inId;
    long long secOutId;
    Vector inputGrad;
    Vector inputVec;
    Vector outputVec;
    Vector secOutVec;
    Vector sumOutVec;
    double lossFirst;
    double lossSecond;
    long long threadTokenCount;
    double lr;
    double inNorm;
    GradManager(int dim, int seed);
    void setLr(double lr);
    double getAverLossFirst();
    double getAverLossSecond();
};

#endif