#include <iostream>
#include <opencv2/opencv.hpp>
#include <string.h>
#include "GMM.h"

using namespace cv;
using namespace std;

void showResult(Mat img1, Mat img2){
    int width = img1.cols + img2.cols;
    int height = img1.rows;
    Mat show = Mat::zeros(Size(width, height), img1.type());
    Rect r1(0, 0, img1.cols, img1.rows);
    Rect r2(0, 0, img2.cols, img2.rows);
    r2.x = img1.cols;
    img1.copyTo(show(r1));
    img2.copyTo(show(r2));
    namedWindow("result", WINDOW_FREERATIO);
    imshow("result", show);
    waitKey(200);
}

string getPath(string base_path, int count){
    string path;
    if(count < 10)
        path = base_path + "b0000" + to_string(count) + ".bmp";
    else if(count >= 10 && count < 100)
        path = base_path + "b000" + to_string(count)  + ".bmp";
    else
        path = base_path + "b00" + to_string(count)  + ".bmp";

    return path;
}

int main() {
    int train_num = 200, test_num = 187, total_num = 287;
    Mat frame, mask, dst;
    GMM gmm;
    string base_path = "/Users/lansy/CLionProjects/GMM_foreground_detection/WavingTrees/";
    int count = 0;
    while(count < train_num) {
        string img_path;
        img_path = getPath(base_path, count);
        frame = imread(img_path, 0);

        if (count == 0) {
            gmm.init(frame);
            cout << "Train frames ......... " << train_num << endl;
        } else
            gmm.train(frame);
        count++;
    }
    cout << "Complete training ..... " << endl;
    gmm.getB(frame.rows, frame.cols);
    cout << "Testing ............... " << endl;

    count = 0;
    while(count < total_num){
        string img_path;
        img_path = getPath(base_path, count);
        Mat img = imread(img_path, 1);
        frame = imread(img_path, 0);
        gmm.test(frame);
        mask = gmm.getMask();
        morphologyEx(mask, mask, MORPH_OPEN, Mat());
        showResult(frame, mask);
        count ++;
    }
    /*
    while(count < total_num){
        string img_path;
        img_path = getPath(base_path, count);
        frame = imread(img_path, 0);

        if(count == 0){
            gmm.init(frame);
            cout << "Train frames ......... " << train_mum << endl;
        }
        else if(count < train_mum){
            gmm.train(frame);
        }
        if(count == train_mum){
            cout << "Complete training ..... " << endl;
            gmm.getB(frame.rows, frame.cols);
            cout << "Testing ............... " << endl;
            cout << "Test frames ........... " << test_num << endl;
        }
        //if(count >= train_mum && count < total_num){
        if(count == 247){
            gmm.test(frame);
            mask = gmm.getMask();
            morphologyEx(mask, mask, MORPH_OPEN, Mat());
            showResult(frame, mask);
        }
        count ++;
    }*/
    return 0;
}