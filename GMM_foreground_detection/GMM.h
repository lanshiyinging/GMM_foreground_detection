//
// Created by lansy on 2019/10/25.
//
#ifndef GMM_FOREGROUND_DETECTION_GMM_H
#define GMM_FOREGROUND_DETECTION_GMM_H

#include <iostream>
#include <cmath>
#include <vector>
#include "opencv2/opencv.hpp"
#define GAUSSIAN_NUM 3
using namespace cv;
using namespace std;

class GMM {
public:
    float init_sd = 18;
    float threshold = 0.75;
    float alpha = 0.05;
    float D = 2;
    Mat weight[GAUSSIAN_NUM];
    Mat mean[GAUSSIAN_NUM];
    Mat sd[GAUSSIAN_NUM];  //standard deviation
    Mat B;
    Mat mask;

public:
    void init(Mat firstFrame);
    void train(Mat frame);
    void getB(int rows, int cols);
    void test(Mat frame);
    Mat getMask();
};


#endif //GMM_FOREGROUND_DETECTION_GMM_H
