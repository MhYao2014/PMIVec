//
// Created by mhyao on 2020/2/6.
//
#ifndef _LOSS_H_
#define _LOSS_H_
#include "Loss.h"
#endif

#ifndef ASSVEC_ADMM_MPI_LOSSFACTORY_H
#define ASSVEC_ADMM_MPI_LOSSFACTORY_H

#include "args.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>


class LossFactory {
private:
    std::string _hard_ware;
    std::string _loss_name;
    bool _lib_can_load;

public:
    explicit LossFactory();
    void setLossName(std::string &loss_name);
    Loss *createLoss();
};

#endif