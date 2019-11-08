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
        weight[k] = Mat::zeros(firstFrame.size(), CV_32FC3);
        mean[k] = Mat::zeros(firstFrame.size(), CV_8UC3);
        sd[k] = Mat::zeros(firstFrame.size(), CV_32FC3);
        if(k == 0){
            weight[k].setTo(1.0);
            /*
            for(int i = 0; i < firstFrame.rows; ++ i){
                for(int j = 0; j < firstFrame.cols; ++ j){
                    mean[k].at<Vec3b>(i, j)[0] = firstFrame.at<Vec3b>(i, j)[0];
                    mean[k].at<Vec3b>(i, j)[1] = firstFrame.at<Vec3b>(i, j)[1];
                    mean[k].at<Vec3b>(i, j)[2] = firstFrame.at<Vec3b>(i, j)[2];
                }
            }
             */
            firstFrame.copyTo(mean[k]);
        }
        else{
            weight[k].setTo(0.0);
            mean[k].setTo(0);
        }
        sd[k].setTo(init_sd);
    }
    B = Mat::ones(firstFrame.size(), CV_8UC3);
    mask = Mat::zeros(firstFrame.size(), CV_8UC3);
}


void GMM::train(Mat frame) {
    float p;
    for(int i = 0; i < frame.rows; ++ i){
        for(int j = 0; j < frame.cols; ++ j){
            for(int n = 0; n < 3; ++ n){
                int match = 0;
                float sum_wight = 0;
                /// calculate the diff between each new pixel and mean
                /// update gaussian model
                for(int k = 0; k < GAUSSIAN_NUM; ++ k){
                    int diff = abs(frame.at<Vec3b>(i,j)[n]- mean[k].at<Vec3b>(i,j)[n]);
                    if(diff <= D*sd[k].at<Vec3f>(i, j)[n]){
                        match = 1;
                        p = alpha / weight[k].at<Vec3f>(i, j)[n];
                        weight[k].at<Vec3f>(i, j)[n] = (1-alpha)*weight[k].at<Vec3f>(i, j)[n] + alpha;
                        mean[k].at<Vec3b>(i,j)[n] =(uchar)((1-p) * mean[k].at<Vec3b>(i,j)[n] + p * frame.at<Vec3b>(i,j)[n]);
                        sd[k].at<Vec3f>(i, j)[n] = (float)sqrt((1-p)*pow(sd[k].at<Vec3f>(i, j)[n],2) + p*pow(frame.at<Vec3b>(i,j)[n]-mean[k].at<Vec3b>(i,j)[n], 2));
                    }
                    else{
                        weight[k].at<Vec3f>(i, j)[n] = (1-alpha)*weight[k].at<Vec3f>(i, j)[n];
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
                            weight[k].at<Vec3f>(i, j)[n] = 1.0 / GAUSSIAN_NUM;
                            mean[k].at<Vec3b>(i, j)[n] = frame.at<Vec3b>(i, j)[n];
                            sd[k].at<Vec3f>(i, j)[n] = init_sd;
                            break;
                        }
                    }
                    //weight[GAUSSIAN_NUM-1].at<float>(i, j) = alpha;
                    if(k >= GAUSSIAN_NUM){
                        mean[GAUSSIAN_NUM-1].at<Vec3b>(i, j)[n] = frame.at<Vec3b>(i, j)[n];
                        sd[GAUSSIAN_NUM-1].at<Vec3f>(i, j)[n] = init_sd;
                    }
                }

                for(int k = 0; k < GAUSSIAN_NUM; ++ k){
                    sum_wight += weight[k].at<Vec3f>(i,j)[n];
                }

                /// normalize weight
                for(int k = 0; k < GAUSSIAN_NUM; ++ k){
                    weight[k].at<Vec3f>(i,j)[n] /= sum_wight;
                }

                /// sort gaussian model
                vector<Rank> ranks;
                float t_w[GAUSSIAN_NUM];
                int t_m[GAUSSIAN_NUM];
                float t_sd[GAUSSIAN_NUM];
                Rank rank;
                for(int k = 0; k < GAUSSIAN_NUM; ++ k){
                    rank.data = weight[k].at<Vec3f>(i, j)[n] / sd[k].at<Vec3f>(i, j)[n];
                    rank.idx = k;
                    ranks.push_back(rank);
                    t_w[k] = weight[k].at<Vec3f>(i, j)[n];
                    t_m[k] = mean[k].at<Vec3b>(i, j)[n];
                    t_sd[k] = sd[k].at<Vec3f>(i, j)[n];
                }
                sort(ranks.begin(), ranks.end(), rule);
                int idx;
                for(int k = 0; k < GAUSSIAN_NUM; ++ k) {
                    idx = ranks[k].idx;
                    weight[k].at<Vec3f>(i, j)[n] = t_w[idx];
                    mean[k].at<Vec3b>(i, j)[n] = (uchar) t_m[idx];
                    sd[k].at<Vec3f>(i, j)[n] = t_sd[idx];
                }
                mask.at<Vec3b>(i, j)[n] = 0;
            }
        }
    }

}

void GMM::getB(int rows, int cols) {

    for(int i = 0; i < rows; ++ i){
        for(int j = 0; j < cols; ++ j ){
            for(int n = 0; n < 3; ++ n){
                float sum_w = 0.0;
                for(int k = 0; k < GAUSSIAN_NUM; ++ k){
                    sum_w += weight[k].at<Vec3f>(i, j)[n];
                    if(sum_w > threshold){
                        B.at<Vec3b>(i,j)[n] = (uchar)(k+1);
                        break;
                    }

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
            for(int n = 0; n < 3; ++ n){
                for(int k = 0; k < B.at<Vec3b>(i, j)[n]; ++ k){
                    int diff = abs(frame.at<Vec3b>(i,j)[n] - mean[k].at<Vec3b>(i,j)[n]);
                    if( diff < D*sd[k].at<Vec3f>(i,j)[n]){
                        match += 1;
                        break;
                    }
                }

            }

            if(match == 3) {
                mask.at<Vec3b>(i, j)[0] = 0;
                mask.at<Vec3b>(i, j)[1] = 0;
                mask.at<Vec3b>(i, j)[2] = 0;
            }
            else{
                mask.at<Vec3b>(i, j)[0] = 255;
                mask.at<Vec3b>(i, j)[1] = 255;
                mask.at<Vec3b>(i, j)[2] = 255;
            }
        }
    }
}

Mat GMM::getMask() {
    return mask;
}
