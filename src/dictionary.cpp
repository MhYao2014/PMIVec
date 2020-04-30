//
// Created by mhyao on 2020/2/25.
//
#pragma once

#include "dictionary.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <thread>

int scmp(char *s1, char *s2) {
    // 以s1字符串为标准，看s2是不是等于s1
    // 两个字符串不相同时，返回1；a=0x7ffffcdff010, b=0x7ffffcdff020
    // 相同时，返回0。
    while (*s1 != '\0' && *s1 == *s2) {s1++;s2++;}
    return *s1 - *s2;
}

int CompareVocab(const void *a, const void *b) {
    long long c;
    if ( (c = ((ARRAYUNIT *)b)->Count - ((ARRAYUNIT *)a)->Count) != 0 ) {
        return c > 0 ? 1 : -1;
    } else {
        return 0;
    }
}

int CompareVocabTie(const void *a, const void *b) {
    long long c;
    if ( (c = ((ARRAYUNIT *) b)->Count - ((ARRAYUNIT *) a)->Count) != 0) return ( c > 0 ? 1 : -1 );
    else return (scmp(((ARRAYUNIT *) a)->Word,((ARRAYUNIT *) b)->Word));
}

Dictionary::Dictionary():_min_count(100),
                        _max_vocab(100000),
                        _if_save_vocab(1),
                        _real_vocab_size(0),
                        _total_tokens(0),
                        vocabArray(NULL),
                        vocabHash(NULL){}

void Dictionary::setCorpusPath(std::string &corpusPath) {
    _corpus_path = corpusPath;
}

long long Dictionary::getTotalTokens() {
    return _total_tokens;
}

Dictionary::~Dictionary() {
    // 不会写析构函数。不知道写的对不对。
    HASHUNITID *htmp= NULL;
    for (int i=0;i<TSIZE;i++) {
        if (vocabHash[i] != NULL) {
            htmp = vocabHash[i];
            while (htmp != NULL) {
                free(htmp->Word);
                htmp->Word = NULL;
                htmp->Count = 0;
                htmp = htmp->next;
            }
        }
    }
    htmp = NULL;
    free(vocabHash);
    free(vocabArray);
}

void Dictionary::setMinCount(long long minCount) {
    _min_count = minCount;
}

void Dictionary::setMaxVocab(long long maxVocab) {
    _max_vocab = maxVocab;
}

void Dictionary::setIfSaveVocab(int ifSaveVocab) {
    _if_save_vocab = ifSaveVocab;
}

long long Dictionary::getRealVocabSize() {
    return _real_vocab_size;
}

void Dictionary::buildVocab() {
    // 在堆上申请内存空间，并初始化指针列表
    vocabHash = (HASHUNITID**) malloc(sizeof(HASHUNITID*) * TSIZE);
    for (int i=0;i<TSIZE;i++) {
        vocabHash[i] = (HASHUNITID *) NULL;
    }
    // 变量定义区，第一次阅读可以跳过，
    // 后面用到再反过来看
    int ifNotGet=1;
    char word[MaxWordLen];
    long long tokenCounter=0;
    long long vocabSize = 1717500;
    vocabArray = (ARRAYUNIT *)malloc(sizeof(ARRAYUNIT) * vocabSize);
    // 打开语料文件并转化为FILE指针
    FILE *corpusFile = fopen(_corpus_path.c_str(),"r");
    fprintf(stderr, "Building Vocabulary.");
    // 逐个读入corpusFile中的每个字符，
    // 并将它们初步压入一个哈希表中，
    // 哈希冲突则将单词使用链表挂在后面。
    while (!feof(corpusFile)) {
        ifNotGet = Dictionary::getWord(corpusFile, word);
        if (ifNotGet) { continue;}
        Dictionary::hashMapWord(word);
        if (((++tokenCounter) % 100000) == 0) {
            fprintf(stderr, "\rHave read %lld tokens so far.",tokenCounter);
        }
    }
    // 记录原始语料有多少单词。
    _total_tokens = tokenCounter;
    // 将带链表的哈希表转化为数组array，方便后面排序
    vocabSize = Dictionary::hashToArray(vocabSize);
//    vocabSize = 718303;
    fprintf(stderr, "\nCounted %lld unique words.\n", vocabSize);
    // 开始利用最大词表长度以及最小词频限制删减词典,并将词表保存在一个文件中
    Dictionary::cutVocab(vocabSize);
    // 根据VocabArray再给哈希表里每个单词赋予id：
    Dictionary::fillIdToVocabHash();
    // 释放内存，防止出现野指针与内存泄漏。
    // 把被砍掉的词汇（表现为id=-1）对应的字符串释放掉，但是指针本身还得保存，方便后续哈希查找。
    HASHUNITID *htmp= NULL;
    for (long long i=0;i<TSIZE;i++) {
        if (vocabHash[i] != NULL) {
            htmp = vocabHash[i];
            while (htmp != NULL) {
                if (htmp->id == -1) {
                    free(htmp->Word);
                    htmp->Word = NULL;
                    htmp->Count = 0;
                }
                htmp = htmp->next;
            }
        }
    }
    htmp = NULL;
}

unsigned int Dictionary::hashValue(char *word, int tsize, unsigned int seed) {
    char c;
    unsigned int h;
    h = seed;
    for ( ; (c = *word) != '\0'; word++) h ^= ((h << 5) + c + (h >> 2));
    return (unsigned int)((h & 0x7fffffff) % tsize);
}

int Dictionary::getWord(FILE *CorpusFile, char *Word) {
    /*
     * 该函数会读取CorpusFile文件指针所指向的文件中的“单个字符串”，
     * 并将这个字符串存在Word字符数组中。
     * “单个字符串”之间的间隔为：\n（转行），
     *                      ' '(空格)，
     *                      \t(制表符)，
     *                      EOF(文件末尾标识)
     *                      等特殊字符。
     */
    int i = 0,ch;
    for (;;) {
        ch = fgetc(CorpusFile);
        if (ch == '\r') continue;
        if (i == 0 && ((ch == '\n') || (ch ==EOF) || (ch == '\0') || (ch == '\v') || (ch =='\f'))) {
            Word[i] = 0;
            return 1;
        }
        if (i == 0 && ((ch == ' ') || (ch == '\t'))) {
            continue;
        }
        if (i != 0 && ((ch == EOF) || (ch == ' ') || (ch == '\t') || (ch == '\n') || (ch == '\0') || (ch == '\v') || (ch =='\f'))) {
            if (ch == '\n') {
                ungetc(ch, CorpusFile);
            }
            break;
        }
        if (i < MaxWordLen) {
            // 处理标点符号
            if ((ispunct(ch) || ch == '\'') && i!=0){
                ungetc(ch, CorpusFile);
                Word[i] = 0;
                return 0;
            }
            if ((ispunct(ch)) && i == 0) {
                Word[i++] = ch;
                Word[i] = 0;
                return 0;
            }
            // 大写改成小写
            if (ch > 64 && ch < 91) {
                ch += 32;
            }
            Word[i++] = ch;
        }
    }
    Word[i] = 0;
    // avoid truncation destroying a multibyte UTF-8 char except if only thing on line (so the i > x tests won't overwrite word[0])
    // see https://en.wikipedia.org/wiki/UTF-8#Description
    if (i == MaxWordLen - 1 && (Word[i-1] & 0x80) == 0x80) {
        if ((Word[i-1] & 0xC0) == 0xC0) {
            Word[i-1] = '\0';
        } else if (i > 2 && (Word[i-2] & 0xE0) == 0xE0) {
            Word[i-2] = '\0';
        } else if (i > 3 && (Word[i-3] & 0xF8) == 0xF0) {
            Word[i-3] = '\0';
        }
    }
    return 0;
}

int Dictionary::getNumInt(FILE *p2File, long long &numInt) {
    /*
     * 假设p2File指向的文件中没有标点符号和小数点
     * numInt会被强制初始化为0
     */
    numInt = 0;
    int i = 0,ch;// i用来标识数字字符串是否已经读过
    for (;;) {
        ch = fgetc(p2File);
        // 文件开头空白跳过
        if (ch == '\r') continue;
        // 直接遇到遇到转行、文件末尾等情况，直接函数返回1，表示什么数字字符串都没读到
        if (i == 0 && ((ch == '\n') || (ch == EOF) || (ch == '\0') || (ch == '\v') || (ch == '\f'))) {
            return 1;
        }
        // 直接遇到空格或者制表符，跳过不读
        if (i == 0 && ((ch == ' ') || (ch == '\t'))) {
            continue;
        }
        // 读取单个字符的过程中遇到各种字符串间隔符号，中断读取过程
        if (i != 0 && ((ch == EOF) || (ch == ' ') || (ch == '\t') || (ch == '\n') || (ch == '\0') || (ch == '\v') ||
                       (ch == '\f'))) {
            if (ch == '\n') {
                ungetc(ch, p2File);
            }
            break;
        }
        // 读取单个字符的过程
        if (i < MaxWordLen) {
            // 将读取到的字符数字转化为整数
            if (ch >= '0' && ch <= '9') {
                numInt = numInt*10 + (ch - '0');
            }
            i++;
        }
    }
    return 0;
}

int Dictionary::getNumEachLine(FILE *p2File,
                                long long &id,
                                int NotReadSuccess,
                                std::vector<long long> &tempWinSamp) {
    tempWinSamp.clear();
    while (true) {
        // getNumInt自己会每次把id归0
        NotReadSuccess = Dictionary::getNumInt(p2File, id);
        if (NotReadSuccess) {
            // 读完了,返回
            return 0;
        } else {
            tempWinSamp.push_back(id);
        }
    }
}

void Dictionary::hashMapWord(char *Word) {
    HASHUNITID *hpre= NULL, *hnow= NULL;
    // 首先计算Word字符串的哈希值
    unsigned int hval = HASHFN(Word, TSIZE, SEED);
    // 然后检查是否哈希冲突(当前是否为空，是否和Word内容一致)，
    // 若冲突则用链表解决:指向链表的下一个单元。
    for (   hpre = NULL,hnow = vocabHash[hval];
            hnow != NULL && scmp(hnow->Word, Word)!=0;
            hpre = hnow, hnow = hnow->next );
    // 此时hnow要么为空，要么和Word内容相同，
    // 和Word内容相同时，那就让hnow指向的结构体中的Count+1。
    // 当hnow为空的时候就新开一个小内存，并把这块小内存的地址给hnow,
    // 最后把Word的内容给这块新的内存，Count+1,next设为NULL(让这块新内存变为链表的最后一个节点)。
    if (hnow == NULL) {
        // hnow为空表明该指针没有指向任何内存，所以需要先开辟内存
        // 然后把hnow指向这个内存。
        hnow = (HASHUNITID *) malloc(sizeof(HASHUNITID));
        hnow->Word = (char *) malloc(strlen(Word) + 1);
        strcpy(hnow->Word, Word);
        hnow->Count = 1;
        hnow->id = -1;
        hnow->next = NULL;
        if (hpre == NULL) {
            // hnow指向的那块内存是VocabHash[hval]处的第一个节点
            vocabHash[hval] = hnow;
        } else {
            // 将hnow接到hpre后面
            hpre->next = hnow;
        }
    } else {
        // hnow不为空，就说明遇到同一个词了
        // 将该词Count+1，同时将hnow指向的内存挂在链表的第一个节点
        hnow->Count++;
        // 先判断hnow所指向的内存在链表中是不是已经处在第一个节点了？
        // 如果是的话(表现为hpre指向NULL)，那就不用再移动了;
        // 如果不是的话(表现为hpre不指向NULL)，那hnow就是处在链表中间位置。
        if (hpre != NULL) {
            hpre->next = hnow->next;
            hnow->next = vocabHash[hval];
            vocabHash[hval] = hnow;
        }
    }
    hnow = NULL;
    hpre = NULL;
}

long long Dictionary::hashToArray(long long vocabSize) {
    // 变量定义区，第一次阅读可以跳过，
    // 后面用到再反过来看
    HASHUNITID *htmp= NULL;
    long long arrayCounter=0;
    // 开始遍历VocabHash的每一行以及每一行中的所有链表
    for (long long HashRow=0;HashRow < TSIZE;HashRow++) {
        htmp = vocabHash[HashRow];
        // 只要htmp不为空，就需要把这一行的所有链表并入数组中
        // 因为这里只是字符串指针的赋值，不需要再开辟内存并拷贝，所以很快
        while (htmp != NULL) {
            vocabArray[arrayCounter].Word = htmp->Word;
            vocabArray[arrayCounter].Count = htmp->Count;
            arrayCounter++;
            // 如果空间不够了，还得再开辟新的内存
            if (arrayCounter >= vocabSize) {
                vocabSize += 2500;
                vocabArray = (ARRAYUNIT *) realloc(vocabArray, sizeof(ARRAYUNIT) * vocabSize);
            }
            htmp = htmp->next;
        }
    }
    // 这里似乎是不需要再赋一次空指针，
    // 因为当初构造链表的时候保证了链表末尾一定是空指针。
    // Anyway，再写一次也无妨。
    htmp = NULL;
    return arrayCounter;
}

void Dictionary::cutVocab(long long VocabSize) {
    // 先砍掉超出最大词汇表长度的单词
    //fprintf(stderr, "\nMaxVocab: %lld; arrarlen: %lld\n", MaxVocab, VocabSize);
    if (_max_vocab > 0 && _max_vocab < VocabSize)
        qsort(vocabArray, VocabSize, sizeof(ARRAYUNIT), CompareVocab);
    else _max_vocab = VocabSize;
    // CompareVocabTie 按照单词首字母的大小裁定两个词频一样的单词谁先谁后
    qsort(vocabArray, _max_vocab, sizeof(ARRAYUNIT), CompareVocabTie);
    FILE* VocabFile = fopen("vocab.txt", "w");
    // 从临界词频（刚刚比MinCount大的词频）处截断词表
    for (long long FinalVocabSize=0; FinalVocabSize < _max_vocab; FinalVocabSize++) {
        if (vocabArray[FinalVocabSize].Count < _min_count) {
            // 将词频不够的单词对应的词字符串指针设为空，count清零。
            vocabArray[FinalVocabSize].Word = NULL;
            vocabArray[FinalVocabSize].Count = 0;
        } else if (_if_save_vocab) {
            _real_vocab_size += 1;
            fprintf(VocabFile,"%s %lld\n", vocabArray[FinalVocabSize].Word, vocabArray[FinalVocabSize].Count);
        }
    }
    fclose(VocabFile);
    // 将超过最大词汇表长度部分的词串指针设为空，count清零。
    if (_max_vocab < VocabSize) {
        for (long long FinalVocabSize=_max_vocab; FinalVocabSize < VocabSize; FinalVocabSize++) {
            vocabArray[FinalVocabSize].Word = NULL;
            vocabArray[FinalVocabSize].Count = 0;
        }
    }
}

void Dictionary::fillIdToVocabHash() {
    // 遍历VocabArray，查找其中的每个单词在VocabHash中的位置
    // 将词汇按词频从大到小的顺序编号并存入哈希表中，从0开始编号。
    // 计算当前单词的hash value，这里不采用传入词汇表长度，
    // 而是另外重新计数是为了让这个函数用起来更加self-contained。更加方便。
    long long VocabSize = 0;
    unsigned int hval;
    HASHUNITID *hnow= NULL;
    while (vocabArray[VocabSize].Word != NULL){
        VocabSize += 1;
    }
    for (long long i=0; i < VocabSize; i++) {
        hval = Dictionary::hashValue(vocabArray[i].Word, TSIZE, SEED);
        hnow = vocabHash[hval];
        while (hnow != NULL) {
            if (scmp(hnow->Word,vocabArray[i].Word) == 0) {
                hnow->id = i;
                break;
            } else {
                hnow = hnow->next;
            }
        }
    }
    hnow = NULL;
}

void Dictionary::setGroups(std::vector<GSIZE> &Groups) {
    GSIZE temp;
    for (int i = 0; i < Groups.size(); i++) {
        temp.TotalSize = Groups[i].TotalSize;
        temp.FileNum = Groups[i].FileNum;
        for (int j = 0; j < Groups[i].FileNames.size(); j++) {
            temp.FileNames.push_back(Groups[i].FileNames[j]);
        }
        groups.push_back(temp);
        temp.FileNames.clear();
    }
}

long long Dictionary::hashSearch(char * Word) {
    // 如果找不到就返回-1
    // 如果找得到就返回单词id值
    long long id = -1;
    unsigned int hval = Dictionary::hashValue(Word, TSIZE, SEED);
    HASHUNITID *htmp = NULL;
    htmp = vocabHash[hval];
    while (htmp != NULL) {
        if (htmp->Word != NULL && scmp(htmp->Word, Word) == 0) {
            id = htmp->id;
            break;
        } else {
            htmp = htmp->next;
        }
    }
    return id;
}

long long Dictionary::getWordId(char * Word) {
    long long id = hashSearch(Word);
    return id;
}

void Dictionary::getLine(FILE *CorpusSplit, std::vector<long long> &line) {
    line.clear();
    line.shrink_to_fit();
    char Word[MaxWordLen];
    int NotReadSuccess;
    long long id;
    while (true) {
        NotReadSuccess = Dictionary::getWord(CorpusSplit, Word);
        if (NotReadSuccess) {
            return;
        } else {
            id = Dictionary::getWordId(Word);
            // 这里没有查到的词也放进去，
            // 稀疏一下line的容量（因为后面会跳过这些词）
            line.push_back(id);
        }
    }
}
