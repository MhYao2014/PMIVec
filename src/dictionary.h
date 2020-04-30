//
// Created by mhyao on 2020/2/25.
//
#ifndef ASSVEC_ADMM_MPI_DICTIONARY_H
#define ASSVEC_ADMM_MPI_DICTIONARY_H

#define TSIZE 1048576
#define SEED 1159241
#define MaxWordLen 1000
#define HASHFN hashValue

#include <iostream>
#include <stdlib.h>
#include <memory>
#include <string>
#include <vector>

#include "args.h"
//#include "FileToWriteSync.h"

typedef struct VocabHashWithId{
    char *Word;
    long long Count;
    long long id;
    struct VocabHashWithId *next;
} HASHUNITID;

typedef struct VocabUnit {
    char * Word;
    long long Count;
} ARRAYUNIT;

typedef struct FileGroup {
    long long TotalSize;
    std::vector<long long> FileNames;
    long long FileNum;
} GSIZE;

typedef struct FileSizeUnit {
    long long FileSize;
    long long FileName;
} FSIZE;

class Dictionary {
private:
    std::string _corpus_path;
    long long _min_count;
    long long _max_vocab;
    int _if_save_vocab;
    long long _real_vocab_size;
    unsigned int hashValue(char *word, int tsize, unsigned int seed);
    void hashMapWord(char *Word);
    int getWord(FILE *corpusFile, char *word);
    long long hashToArray(long long vocabSize);
    void cutVocab(long long vocabSize);
    void fillIdToVocabHash();
    long long getWordId(char *Word);
    long long hashSearch(char *Word);
    //    long long hashToArray(HASHUNITID **vocabHash, ARRAYUNIT * vocabArray, long long vocabSize);
    //    void cutVocab(ARRAYUNIT* vocabArray, long long vocabSize);
    //    void hashMapWord(char *Word, HASHUNITID **VocabHash);
    //    void fillIdToVocabHash(ARRAYUNIT *vocabArray, HASHUNITID ** vocabHash);
public:
    long long _total_tokens;
    HASHUNITID ** vocabHash;
    ARRAYUNIT * vocabArray;
    // todo:修缮splitcorpus成员函数
    std::vector<GSIZE> groups;
    explicit Dictionary();
    ~Dictionary();
    void setCorpusPath(std::string &corpusPath);
    void setMinCount(long long minCount);
    void setMaxVocab(long long maxVocab);
    void setIfSaveVocab(int ifSaveVocab);
    void setGroups(std::vector<GSIZE> &Groups);
    long long getRealVocabSize();
    void buildVocab();
    void getLine(FILE * CorpusSplit, std::vector<long long> &line);
    long long getTotalTokens();

    int getNumInt(FILE *p2File, long long &numInt);
    int getNumEachLine(FILE * p2File,
                    long long &numInt,
                    int NotReadSuccess,
                    std::vector<long long> &WinSamp);
};

#endif