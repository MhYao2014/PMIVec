#ifndef SECVEC_SECVEC_H
#define SECVEC_SECVEC_H

#include <atomic>
#include <chrono>
#include <thread>
#include "Loss.h"
#include "dictionary.h"

class SecVec: public Loss {
private:
    int dim;
    long long vocabSize;
    double process;
    std::atomic<double> lossFirst;
    std::atomic<double> lossSecond;
    std::atomic<long long> allThreadToken;
    std::shared_ptr<Matrix> p2Input;
    std::shared_ptr<Matrix> p2Output;
    void initNegAndUniTable(Dictionary* p2Dict);
    long long pickSecOutId(int lB, int rB, int seed, int lineIndex,int shift, std::vector<long long>& line);
    void saveVec(Dictionary* p2Dict,Args* p2Args);
    void printInfo(double process, double lossFirst, double lossSecond, std::ostream& log_stream);

public:
    SecVec();
    void initVariables(Dictionary* p2Dict, Args* p2Args) override;
    void train(Dictionary* p2Dict, Args* p2Args) override;
    void eachThread(int threadId, Dictionary* p2DIct, Args* p2Args);
    void forward(long long outId, long long secOutId, GradManager& gradient, double& tmpLossFirst, double& tmpLossSecond, Args* p2Args);
    void biLogiFirst(long long outId,
                    GradManager& gradient,
                    bool lableIsPositive,
                    double& tmpLossFirst);
    void biLogiSecond(long long outId,
                      long long secOutId,
                      GradManager& gradient,
                      bool lableIsPositive,
                      double& tmpLossSecond);
    void lossEachLine(int threadId,
            Dictionary* p2Dict,
            Args* p2Args,
            std::vector<long long> line,
            GradManager& gradient);
    void gradUpdate(GradManager& gradient, Args* p2Args);
    double shrinkLr(GradManager& gradient, Dictionary* p2Dict, Args* p2Args);
};


#endif //SECVEC_SECVEC_H
