//
// Created by lansy on 2019/10/25.
//

#include "GMM.h"


struct Rank{
    float data;
    int idx;
};

bool rule(Rank r1, Rank r2){
    return r1.data > r2.data;
}

void GMM::init(Mat firstFrame) {
    /// init gaussian mixture model for each pixel with first frame
    /** The weight of first gaussian model set to 1, the others set to 0
     * The mean of first gaussian model set to first frame value, the others set to 0
     * The sigma of each gaussian model set to 15
    */
    for(int k = 0; k < GAUSSIAN_NUM; ++ k){
        weight[k] = Mat::zeros(firstFrame.size(), CV_32FC1);
        mean[k] = Mat::zeros(firstFrame.size(), CV_8UC1);
        sd[k] = Mat::zeros(firstFrame.size(), CV_32FC1);
        if(k == 0){
            weight[k].setTo(1.0);
            firstFrame.copyTo(mean[k]);
        }
        else{
            weight[k].setTo(0.0);
            mean[k].setTo(0);
        }
        sd[k].setTo(init_sd);
    }
    B = Mat::ones(firstFrame.size(), CV_8UC1);
    mask = Mat::zeros(firstFrame.size(), CV_8UC1);
}


void GMM::train(Mat frame) {
    float p;
    for(int i = 0; i < frame.rows; ++ i){
        for(int j = 0; j < frame.cols; ++ j){
            int match = 0;
            float sum_wight = 0;
            /// calculate the diff between each new pixel and mean
            /// update gaussian model
            for(int k = 0; k < GAUSSIAN_NUM; ++ k){
                int diff = abs(frame.at<uchar>(i,j) - mean[k].at<uchar>(i,j));
                if(diff <= D*sd[k].at<float>(i, j)){
                    match = 1;
                    p = alpha / weight[k].at<float>(i, j);
                    weight[k].at<float>(i, j) = (1-alpha)*weight[k].at<float>(i, j) + alpha;
                    mean[k].at<uchar>(i,j) =(uchar)((1-p) * mean[k].at<uchar>(i,j) + p * frame.at<uchar>(i,j));
                    sd[k].at<float>(i, j) = (float)sqrt((1-p)*pow(sd[k].at<float>(i, j),2) + p*pow(frame.at<uchar>(i,j)-mean[k].at<uchar>(i,j), 2));
                    //sd[k].at<float>(i, j) = sqrt((1-p)*pow(sd[k].at<float>(i, j),2) + p*pow(diff, 2));
                }
                else{
                    weight[k].at<float>(i, j) = (1-alpha)*weight[k].at<float>(i, j);
                }
            }

            /// if no match, create a new gaussian model
            /*
            if(match == 0){
                weight[GAUSSIAN_NUM-1].at<float>(i, j) = alpha;
                mean[GAUSSIAN_NUM-1].at<uchar>(i, j) = frame.at<uchar>(i, j);
                sd[GAUSSIAN_NUM-1].at<float>(i, j) = init_sd;
            }
             */

            if(match == 0){
                int k;
                for(k = 0; k < GAUSSIAN_NUM; ++ k){
                    if(weight[k].at<float>(i, j) == 0){
                        //cout << 1;
                        weight[k].at<float>(i, j) = 1.0 / GAUSSIAN_NUM;
                        mean[k].at<uchar>(i, j) = frame.at<uchar>(i, j);
                        sd[k].at<float>(i, j) = init_sd;
                        break;
                    }
                }
                //weight[GAUSSIAN_NUM-1].at<float>(i, j) = alpha;
                if(k >= GAUSSIAN_NUM){
                    //cout << 2;
                    mean[GAUSSIAN_NUM-1].at<uchar>(i, j) = frame.at<uchar>(i, j);
                    sd[GAUSSIAN_NUM-1].at<float>(i, j) = init_sd;
                }
            }

            for(int k = 0; k < GAUSSIAN_NUM; ++ k){
                sum_wight += weight[k].at<float>(i,j);
            }

            /// normalize weight
            for(int k = 0; k < GAUSSIAN_NUM; ++ k){
                weight[k].at<float>(i,j) /= sum_wight;
            }

            /// sort gaussian model
            vector<Rank> ranks;
            float t_w[GAUSSIAN_NUM];
            int t_m[GAUSSIAN_NUM];
            float t_sd[GAUSSIAN_NUM];
            Rank rank;
            for(int k = 0; k < GAUSSIAN_NUM; ++ k){
                rank.data = weight[k].at<float>(i, j) / sd[k].at<float>(i, j);
                rank.idx = k;
                ranks.push_back(rank);
                t_w[k] = weight[k].at<float>(i, j);
                t_m[k] = mean[k].at<uchar>(i, j);
                t_sd[k] = sd[k].at<float>(i, j);
            }
            sort(ranks.begin(), ranks.end(), rule);
            int idx;
            for(int k = 0; k < GAUSSIAN_NUM; ++ k){
                idx = ranks[k].idx;
                weight[k].at<float>(i, j) = t_w[idx];
                mean[k].at<uchar>(i, j) = (uchar)t_m[idx];
                sd[k].at<float>(i, j) = t_sd[idx];
            }
        }
    }

}

void GMM::getB(int rows, int cols) {

    for(int i = 0; i < rows; ++ i){
        for(int j = 0; j < cols; ++ j ){
            float sum_w = 0.0;
            for(int k = 0; k < GAUSSIAN_NUM; ++ k){
                sum_w += weight[k].at<float>(i, j);
                if(sum_w > threshold){
                    B.at<uchar>(i,j) = (uchar)(k+1);
                    break;
                }

            }
        }
    }
}

void GMM::test(Mat frame) {
    /// get b to determine background
    for(int i = 0; i < frame.rows; ++ i){
        for(int j = 0; j < frame.cols; ++ j){
            int match = 0;
            for(int k = 0; k < B.at<uchar>(i, j); ++ k){
                int diff = abs(frame.at<uchar>(i,j) - mean[k].at<uchar>(i,j));
                if(diff < D*sd[k].at<float>(i,j)){
                    match = 1;
                    mask.at<uchar>(i, j) = 0;
                    break;
                }
            }
            if(match == 0)
                mask.at<uchar>(i, j) = 255;
        }
    }
}

Mat GMM::getMask() {
    return mask;
}
