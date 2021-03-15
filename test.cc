#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <ctime>
#include <iomanip>
#include <typeinfo>
#include <opencv2/opencv.hpp>
#include <Eigen/Array>

using namespace std;

#define HALF_PATCH_SIZE 15
///home/poseidon/Documents/00/image_0/000000.png
int main(int argc, char** argv)
{
    cv::Mat img1 = cv::imread("/home/poseidon/Documents/00/image_0/000000.png", cv::ImreadModes::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread("/home/poseidon/Documents/00/image_0/000001.png", cv::ImreadModes::IMREAD_GRAYSCALE);
    // cv::Mat img1 = cv::imread("/home/poseidon/Downloads/rgbd_dataset_freiburg3_walking_xyz/rgb/1341846313.553992.png", cv::ImreadModes::IMREAD_GRAYSCALE);
    // cv::Mat img2 = cv::imread("/home/poseidon/Downloads/rgbd_dataset_freiburg3_walking_xyz/rgb/1341846313.592026.png", cv::ImreadModes::IMREAD_GRAYSCALE);
    cv::Ptr<cv::ORB> orb =  cv::ORB::create();

    vector<cv::KeyPoint> keypoints_1;
    vector<cv::KeyPoint> keypoints_2;
    orb->detect(img1, keypoints_1);
    orb->detect(img1, keypoints_2);

    cv::drawKeypoints(img1, keypoints_1, img1, cv::Scalar::all(-1));
    cv::drawKeypoints(img2, keypoints_2, img2, cv::Scalar::all(-1));

    cv::Mat descriptor_1;
    cv::Mat descriptor_2;
    orb->compute(img1, keypoints_1, descriptor_1);
    orb->compute(img2, keypoints_2, descriptor_2);

    vector<cv::DMatch> matches;

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(descriptor_1, descriptor_2, matches);

    vector<cv::DMatch> good_matches;
    double max_distance = 0;
    double min_distance = 10000;

    for(int i = 0; i< descriptor_1.rows; i++)
    {
        double dist = matches[i].distance;
        if(dist > max_distance)
        {
            max_distance = dist;
        }
        if(dist < min_distance)
        {
            min_distance = dist;
        }
    }
    cout<<"Max distance: "<<max_distance<<endl;
    cout<<"Min distance: "<<min_distance<<endl;

    for(int i = 0; i<matches.size(); i++)
    {
        if(matches[i].distance < max(2*min_distance, 20.0))
        {
            cv::DMatch match;
            match.queryIdx = matches[i].queryIdx;
            match.trainIdx = matches[i].trainIdx;
            // good_matches.push_back(matches[i]);
            good_matches.push_back(match);
        }
    }
    cout<<matches.size()<<endl;
    cout<<good_matches.size()<<endl;
    vector<cv::DMatch> testMatch;
    testMatch.push_back(good_matches[2]);

    cv::Mat dst;
    cv::drawMatches(img1, keypoints_1, img2, keypoints_2, testMatch, dst);
    // int N = int(good_matches.size());
    // cv::Mat points1(N, 2, CV_32FC1);
    // cv::Mat points2(N, 2, CV_32FC1);
    // for(int i = 0; i<good_matches.size(); i++)
    // {
    //     points1.at<float>(i,0) = keypoints_1[good_matches[i].queryIdx].pt.x;
    //     points1.at<float>(i,1) = keypoints_1[good_matches[i].queryIdx].pt.y;

    //     points2.at<float>(i,0) = keypoints_2[good_matches[i].trainIdx].pt.x;
    //     points2.at<float>(i,1) = keypoints_2[good_matches[i].trainIdx].pt.y;
    // }
    // // cv::Mat H = cv::findHomography(points1, points2);
    // vector<uchar> RANSACStatus;
    // cv::Mat F = cv::findFundamentalMat(points1, points2, RANSACStatus, cv::FM_8POINT);
    // cout<<F<<endl;
    // cv::Point2d center(607.1928,185.2157);
    // double focal_lenght = 718.856;
    // cv::Mat ess = cv::findEssentialMat(points1, points2, focal_lenght, center);
    // cout<<"ess :"<<ess<<endl;
    
    // // cout<<H<<endl;

    // // double fx, fy, cx, cy;
    // cv::Mat K = cv::Mat::eye(3,3, CV_64FC1);
    // K.at<double>(0,0) = 718.856;
    // K.at<double>(0,2) = 607.1928;
    // K.at<double>(1,1) = 718.856;
    // K.at<double>(1,2) = 185.2157;
    // cout<<"K: "<<K<<endl;

    // cv::Mat Kt = K.t();
    // cout<<"Kt: "<<Kt<<endl;

    // cv::Mat E = Kt*F*K;
    // cout<<"E: "<<E<<endl;

    // cv::SVD svd(ess);
    // cv::Mat W = cv::Mat::eye(3,3,CV_64FC1);
    // W.at<double>(0,1) = -1;
    // W.at<double>(1,0) = 1;
    // W.at<double>(2,2) = 1;

    // cv::Mat_<double> R = svd.u*W*svd.vt;
    // cv::Mat_<double> t = svd.u.col(2);
    // cout<<"R: "<<R<<endl;
    // cout<<"t: "<<t<<endl;
    

    // cv::Mat pose1(3,4,CV_64FC1);
    // cv::Mat pose2(3,4,CV_64FC1);
    // string filename = "/home/poseidon/Downloads/evo-master/test/data/KITTI_00_gt.txt";
    // ifstream f;
    // f.open(filename);
    // double d;
    // // getline(f, s);
    // for(int i = 0; i<pose1.rows; i++)
    // {
    //     for(int j = 0; j<pose1.cols; j++)
    //     {
    //         if(!f.eof()) f>>d;
    //         pose1.at<double>(i,j) = d;
    //     }
    // }
    // for(int i = 0; i<pose2.rows; i++)
    // {
    //     for(int j = 0; j<pose2.cols; j++)
    //     {
    //         if(!f.eof()) f>>d;
    //         pose2.at<double>(i,j) = d;
    //     }
    // }
    // cv::Mat pose1_33(3,3,CV_64FC1);
    // cv::Mat pose2_33(3,3,CV_64FC1);
    // cv::Mat pose1_t(3, 1, CV_64FC1);
    // cv::Mat pose2_t(3, 1, CV_64FC1);
    // pose1_33 = pose1.rowRange(0,3).colRange(0,3);
    // pose2_33 = pose1.rowRange(0,3).colRange(0,3);
    // pose1_t = pose1.col(3);
    // pose2_t = pose2.col(3);
    // cout<<"pose1_t: "<<pose1_t<<endl;
    // cout<<"pose2_t: "<<pose2_t<<endl;
    // cv::Mat R_gt = pose2_33*pose1_33.inv();
    // cv::Mat t_gt = pose2_t - pose1_t;
    // cout<<"R gt: "<<R_gt<<endl;
    // cout<<"t gt: "<<t_gt<<endl;
    // cv::Mat t_gt_hat(3, 3, CV_64FC1);
    // t_gt_hat.at<double>(0,1) = -t_gt.at<double>(2,0);
    // t_gt_hat.at<double>(1,0) = t_gt.at<double>(2,0);
    // t_gt_hat.at<double>(0,2) = t_gt.at<double>(1,0);
    // t_gt_hat.at<double>(2,0) = -t_gt.at<double>(1,0);
    // t_gt_hat.at<double>(1,2) = -t_gt.at<double>(0,0);
    // t_gt_hat.at<double>(2,1) = t_gt.at<double>(0,0);
    // cout<<"t_hat: "<<t_gt_hat<<endl;
    // cv::Mat E_gt = t_gt_hat*R_gt;
    // cout<<"E_gt: "<<E_gt<<endl;
    // // R.at<double>(0,0) = 0.99999237;
    // // R.at<double>(0,0) = 0.0024387874;
    // // R.at<double>(0,0) = 0.0030571159;
    // // R.at<double>(0,0) = -0.0024455413;
    // // R.at<double>(0,0) = 0.99999458;
    // // R.at<double>(0,0) = 0.0022075167;
    // // R.at<double>(0,0) = -0.0030517157;
    // // R.at<double>(0,0) = -0.0022149761;
    // // R.at<double>(0,0) = 0.99999291;
    // cv::Mat result = R.inv()*pose2_33;
    // cout<<"result: "<<result<<endl;
    // // cout<<"pose1_33: "<<pose2_33<<endl;
    // cout<<"pose1: "<<pose1<<endl;
    // cout<<"pose2: "<<pose2<<endl;

    // cv::Mat posetest(4,4,CV_64FC1,cv::Scalar(0));
    // // for(int i = 0; i<posetest.rows; i++)
    // // {
    // //     for(int j = 0; j<posetest.cols; j++)
    // //     {
    // //         // if(!f.eof()) f>>d;
    // //         posetest.at<double>(i,j) = i*4 + j;
    // //     }
    // // }
    // posetest.at<double>(0,0) = 1;
    // posetest.at<double>(1,1) = 1;
    // posetest.at<double>(2,2) = 1;
    // posetest.at<double>(0,3) = 1;
    // posetest.at<double>(1,3) = 1;
    // posetest.at<double>(2,3) = 1;
    // posetest.at<double>(3,3) = 1;

    // cout<<"test: "<<endl<<posetest.inv()<<endl;
    // // Tcw: [0.99999237, 0.0024387874, 0.0030571159, 0.00041114329;
    // //     -0.0024455413, 0.99999458, 0.0022075167, 0.00015805273;
    // //     -0.0030517157, -0.0022149761, 0.99999291, -0.040841851;
    // //     0, 0, 0, 1]




    

    // // const int npoints = 512;
    // // const cv::Point* pattern0 = (const cv::Point*)bit_pattern_31_;
    // // vector<cv::Point> pattern;
    // // std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));

    // // std::vector<int> umax;
    // // umax.resize(HALF_PATCH_SIZE+1);
    // // int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
    // // int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    // // const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;
    // // for (v = 0; v <= vmax; ++v)
    // //     umax[v] = cvRound(sqrt(hp2 - v * v));

    // // // Make sure we are symmetric
    // // for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
    // // {
    // //     while (umax[v0] == umax[v0 + 1])
    // //         ++v0;
    // //     umax[v] = v0;
    // //     ++v0;
    // // }
    // // cv::Mat dst;
    // // for(int i = 0; i < img1.rows; i++)
    // // {
    // //     if(i>100 && i<400)
    // //     {
    // //         dst.push_back(img1.row(i));
    // //     }
    // // }
    // // cout<<dst.rows<<endl;

    
    // // cv::imshow("img1", img1);
    cv::imshow("dst", dst);
    // cv::imshow("img2", img2);
    cv::waitKey(0);

    return 0;
}