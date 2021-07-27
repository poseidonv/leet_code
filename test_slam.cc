#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <ctime>
#include <iomanip>
#include <typeinfo>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <cstdlib>

using namespace std;

#define HALF_PATCH_SIZE 15
#define MAX_ITERATIONS 10
///home/poseidon/Documents/00/image_0/000000.png

void GenerateMatches(cv::Mat& img1, vector<cv::KeyPoint>& keypoints1, cv::Mat& img2, vector<cv::KeyPoint>& keypoints2, vector<cv::DMatch>& good_matches);

vector<vector<size_t>> GenerateEightPoints(int N);

void GetTruthPose(cv::Mat& R, cv::Mat& t);

cv::Mat FindFundamental(vector<cv::KeyPoint>& points1, vector<cv::KeyPoint>& points2, vector<cv::DMatch>& good_matches,vector<vector<size_t>>& sets);

void DecomposeE(cv::Mat& E, cv::Mat& R, cv::Mat& t, cv::Mat& K, vector<cv::KeyPoint>& keypoints_1, vector<cv::KeyPoint>& keypoints_2, vector<cv::DMatch>& good_matches, vector<cv::Point3f>& vP3D1);

int CheckRT(cv::Mat& R, cv::Mat& t, cv::Mat& K, vector<cv::KeyPoint>& keypoints_1, vector<cv::KeyPoint>& keypoints_2, vector<cv::DMatch>& good_matches, vector<cv::Point3f>& vP3D1);

void Normalize(vector<cv::KeyPoint>& vKeys, vector<cv::Point2f>& normalizedKeys, cv::Mat& T);

cv::Mat ComputeF21(vector<cv::Point2f>& vPn1i, vector<cv::Point2f>& vPn2i);

float CheckFundamental(vector<cv::KeyPoint>& points1, vector<cv::KeyPoint>& points2, vector<cv::DMatch>& good_matches, cv::Mat& F21, vector<bool>& vbCurrentInliers, float sigma);

void Triangulate(cv::Point2f& kp1, cv::Point2f& kp2, cv::Mat& P1, cv::Mat& P2, cv::Mat& p3dC1);

void GetPredict(vector<cv::KeyPoint>& keypoints_1, vector<cv::KeyPoint>& keypoints_2, vector<cv::KeyPoint>& keypoints_2_hat,vector<cv::DMatch>& good_matches, cv::Mat& R, cv::Mat& t, vector<cv::Point3f>& vP3D1);

int main(int argc, char** argv)
{
    cv::Mat img1 = cv::imread("/home/poseidon/Documents/00/image_0/000000.png", cv::ImreadModes::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread("/home/poseidon/Documents/00/image_0/000001.png", cv::ImreadModes::IMREAD_GRAYSCALE);
    // cv::Mat img1 = cv::imread("/home/poseidon/Downloads/rgbd_dataset_freiburg3_walking_xyz/rgb/1341846313.553992.png", cv::ImreadModes::IMREAD_GRAYSCALE);
    // cv::Mat img2 = cv::imread("/home/poseidon/Downloads/rgbd_dataset_freiburg3_walking_xyz/rgb/1341846313.990058.png", cv::ImreadModes::IMREAD_GRAYSCALE);
    vector<cv::KeyPoint> keypoints_1;
    vector<cv::KeyPoint> keypoints_2;
    vector<cv::DMatch> good_matches;

    GenerateMatches(img1, keypoints_1, img2, keypoints_2, good_matches);
    
    int N = int(good_matches.size());
    vector<cv::Point2f> points1;
    vector<cv::Point2f> points2;
    cout<<"good matches 20: "<<keypoints_1[good_matches[20].queryIdx].pt<<endl;
    cout<<"good matches 20: "<<keypoints_2[good_matches[20].trainIdx].pt<<endl;

    for(int i = 0; i<good_matches.size(); i++)
    {
        points1.push_back(keypoints_1[good_matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[good_matches[i].trainIdx].pt);
        // cout<<i<<"--------------------------------------"<<endl;
        // cout<<keypoints_1[good_matches[i].queryIdx].pt.x<<" "<<keypoints_1[good_matches[i].queryIdx].pt.y<<endl; 
        // cout<<keypoints_2[good_matches[i].trainIdx].pt.x<<" "<<keypoints_2[good_matches[i].trainIdx].pt.y<<endl;
        // cout<<"--------------------------------------"<<endl;
    }

    vector<vector<size_t>> sets = GenerateEightPoints(good_matches.size());

    cv::Mat F = cv::findFundamentalMat(points1, points2, cv::FM_8POINT); 
    cout<<"F: "<<F<<endl;
    cv::Mat F21 = FindFundamental(keypoints_1, keypoints_2, good_matches, sets);
    cout<<"F21: "<<F21<<endl;
    F = F21;
    cv::Point2d center(607.1928,185.2157);
    double focal_lenght = 718.856;
    cv::Mat ess = cv::findEssentialMat(points1, points2, focal_lenght, center);
    cout<<"ess :"<<ess<<endl;

    // double fx, fy, cx, cy;
    cv::Mat K = cv::Mat::eye(3,3, CV_32FC1);
    K.at<float>(0,0) = 718.856;
    K.at<float>(0,2) = 607.1928;
    K.at<float>(1,1) = 718.856;
    K.at<float>(1,2) = 185.2157;
    // cout<<"K: "<<K<<endl;

    cv::Mat Kt = K.t();
    // cout<<"Kt: "<<Kt<<endl;
    F.convertTo(F, CV_32FC1);
    cv::Mat E = Kt*F*K;
    cout<<"E: "<<E<<endl;

    vector<cv::Point3f> vP3D1;
    vP3D1.resize(keypoints_1.size());

    E.convertTo(E, CV_32FC1);
    cv::Mat R, t;
    // cv::Mat E1 = E.clone();
    // E1.convertTo(E1, CV_64FC1);
    // cv::recoverPose(E1, points1, points2, K, R, t);
    // R.convertTo(R, E.type());
    // t.convertTo(t, E.type());
    DecomposeE(E, R, t, K, keypoints_1, keypoints_2, good_matches, vP3D1);
    cout<<"R: \n"<<R<<endl;
    cout<<"t: \n"<<t<<endl;

    cv::Mat R_gt, t_gt;
    GetTruthPose(R_gt, t_gt);
    cout<<"R_gt: \n"<<R_gt<<endl;
    cout<<"t_gt: \n"<<t_gt<<endl;

    cv::Mat point1(3,1,CV_32FC1);
    cv::Mat point2(3,1,CV_32FC1);
    point1.at<float>(0,0) = 218.574;
    point1.at<float>(1,0) = 261.572;
    point1.at<float>(2,0) = 1;
    point2.at<float>(0,0) = 739;
    point2.at<float>(1,0) = 140;
    point2.at<float>(2,0) = 1;
    // cout<<"dot: "<<point1.dot(point2)<<endl;
    // cout<<"norm: "<<cv::norm(point1)<<endl;
    point1.convertTo(point1, E.type());
    cv::Mat point1_hat(3,1,CV_32FC1);
    point1_hat = (R*point1+t);
    cout<<"point1_hat: "<<point1_hat<<endl; 

    vector<cv::KeyPoint> keypoints_2_hat(keypoints_1.size(), cv::KeyPoint());
    // cout<<"R: \n"<<R<<endl;
    // [0.99999607, -0.0028020758, -0.00034713745;
    // 0.0028023217, 0.99999583, 0.00070169196;
    // 0.00034508109, -0.00070266519, 0.99999964]
    // cout<<"t: "<<t<<endl;
    // [0.81548434;
    // 0.018546032;
    // -0.57848185]
    GetPredict(keypoints_1, keypoints_2, keypoints_2_hat, good_matches, R, t, vP3D1);
    cv::Mat dst;
    vector<cv::DMatch> testMatch;
    testMatch.push_back(good_matches[23]);
    // testMatch.push_back(good_matches[6]);
    // testMatch.push_back(good_matches[17]);
    // testMatch.push_back(good_matches[22]);
    // testMatch.push_back(good_matches[23]);

    cv::drawMatches(img1, keypoints_1, img2, keypoints_2_hat, testMatch, dst);
    // cv::drawMatches(img1, keypoints_1, img2, keypoints_2, testMatch, dst);

    vector<cv::KeyPoint> points2_hat;

    for(int i = 0; i<good_matches.size(); i++)
    {
        cv::Mat p1(3,1,CV_32FC1);
        p1.at<float>(0,0) = keypoints_1[good_matches[i].queryIdx].pt.x;
        p1.at<float>(1,0) = keypoints_1[good_matches[i].queryIdx].pt.y;
        p1.at<float>(2,0) = 1;
        cv::Mat p_hat = R*p1+t;
        cv::KeyPoint p2_hat;
        p2_hat.pt.x = p_hat.at<float>(0,0);
        p2_hat.pt.y = p_hat.at<float>(1,0);
        points2_hat.push_back(p2_hat);
        // cout<<"-----------------------"<<endl;
        // cout<<"p1: "<<keypoints_1[good_matches[i].queryIdx].pt<<" kp2: "<<keypoints_2[good_matches[i].trainIdx].pt<<endl;
        // cout<<"p1: "<<keypoints_1[good_matches[i].queryIdx].pt<<" p2_hat: "<<points2_hat[i].pt<<endl;
        // cout<<"-----------------------"<<endl;
        cv::line(img1, keypoints_1[good_matches[i].queryIdx].pt, points2_hat[i].pt, cv::Scalar(0,255,0) );
    }
    cv::imshow("img1", img1);

    // F: [1.977825191996777e-05, 0.001941681404665353, -0.8411690511960619;
    //     -0.001941691206089452, -1.759565030357177e-05, -0.03489866852588643;
    //     0.8302828924955168, 0.03964603470728242, 1]

    // F: [8.902113764578159e-06, 0.00149206613620167, -0.6274287915318005;
    //     -0.001489959699949435, 1.389287271442163e-06, 0.05119984475505795;
    //     0.621691484274663, -0.05378737494078776, 1]

    // cout<<"point2 norm: "<<point2<<endl;
    // cv::Mat r;
    // cv::Rodrigues(R, r);
    // cout<<"r: "<<r<<endl;

    // // cv::imshow("img1", img1);
    cv::imshow("dst", dst);
    // cv::imshow("img2", img2);
    cv::waitKey(0);

    return 0;
}

void GetTruthPose(cv::Mat& R, cv::Mat& t)
{
    cv::Mat pose1(3,4,CV_32FC1);
    cv::Mat pose2(3,4,CV_32FC1);
    string filename = "/home/poseidon/Downloads/evo-master/test/data/KITTI_00_gt.txt";
    ifstream f;
    f.open(filename);
    float x;
    string s;
    // for (int i = 0; i < 53; i++)
    // {
    //     getline(f,s);
    // }
    
    // getline(f, s);
    for(int i = 0; i<pose1.rows; i++)
    {
        for(int j = 0; j<pose1.cols; j++)
        {
            if(!f.eof()) f>>x;
            pose1.at<float>(i,j) = x;
        }
    }
    for(int i = 0; i<pose2.rows; i++)
    {
        for(int j = 0; j<pose2.cols; j++)
        {
            if(!f.eof()) f>>x;
            pose2.at<float>(i,j) = x;
        }
    }
    cv::Mat pose1_33(3,3,CV_32FC1);
    cv::Mat pose2_33(3,3,CV_32FC1);
    cv::Mat pose1_t(3, 1, CV_32FC1);
    cv::Mat pose2_t(3, 1, CV_32FC1);
    pose1_33 = pose1.rowRange(0,3).colRange(0,3);
    pose2_33 = pose1.rowRange(0,3).colRange(0,3);
    pose1_t = pose1.col(3);
    pose2_t = pose2.col(3);
    // cout<<"pose1_t: \n"<<pose1_t<<endl;
    // cout<<"pose2_t: \n"<<pose2_t<<endl;
    cv::Mat R_gt = pose2_33*pose1_33.inv();
    cv::Mat t_gt = pose2_t - pose1_t;
    // cout<<"R gt: \n"<<R_gt<<endl;
    // cout<<"t gt: \n"<<t_gt<<endl;
    cv::Mat t_gt_hat(3, 3, CV_32FC1);
    t_gt_hat.at<float>(0,1) = -t_gt.at<float>(2,0);
    t_gt_hat.at<float>(1,0) = t_gt.at<float>(2,0);
    t_gt_hat.at<float>(0,2) = t_gt.at<float>(1,0);
    t_gt_hat.at<float>(2,0) = -t_gt.at<float>(1,0);
    t_gt_hat.at<float>(1,2) = -t_gt.at<float>(0,0);
    t_gt_hat.at<float>(2,1) = t_gt.at<float>(0,0);
    // cout<<"t_hat: "<<t_gt_hat<<endl;
    cv::Mat E_gt = t_gt_hat*R_gt;
    cout<<"E_gt: "<<E_gt<<endl;
    // cv::Mat result = R.inv()*pose2_33;
    // cout<<"result: "<<result<<endl;
    // cout<<"pose1_33: "<<pose2_33<<endl;
    // cout<<"pose1: "<<pose1<<endl;
    // cout<<"pose2: "<<pose2<<endl;

    R = R_gt;
    t = t_gt;
}

void GenerateMatches(cv::Mat& img1, vector<cv::KeyPoint>& keypoints1, cv::Mat& img2, vector<cv::KeyPoint>& keypoints2, vector<cv::DMatch>& good_matches)
{
    cv::Ptr<cv::ORB> orb =  cv::ORB::create();
    orb->detect(img1, keypoints1);
    orb->detect(img2, keypoints2);

    cv::Mat img1_test = img1.clone();
    cv::cvtColor(img1_test, img1_test, CV_GRAY2BGR);

    cv::circle(img1_test, cv::Point2d(751, 143), 10, cv::Scalar(0,0,255));
    // cv::imshow("tes1", img1_test);

    cv::Mat img2_test = img2.clone();
    cv::cvtColor(img2_test, img2_test, CV_GRAY2BGR);

    cv::circle(img2_test, cv::Point2d(756, 140), 10, cv::Scalar(0,0,255));
    // cv::imshow("test2", img2_test);

    // cv::drawKeypoints(img1, keypoints_1, img1, cv::Scalar::all(-1));
    // cv::drawKeypoints(img2, keypoints_2, img2, cv::Scalar::all(-1));

    cv::Mat descriptor_1;
    cv::Mat descriptor_2;
    orb->compute(img1, keypoints1, descriptor_1);
    orb->compute(img2, keypoints2, descriptor_2);

    vector<cv::DMatch> matches;

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(descriptor_1, descriptor_2, matches);

    // vector<cv::DMatch> good_matches;
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
        if(matches[i].distance < max(2*min_distance, 30.0))
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
}

vector<vector<size_t>> GenerateEightPoints(int N)
{
    vector<size_t> vAllIndices;
    vector<size_t> vAvailableIndices;
    assert(N > 8);
    for(int i = 0; i< N; i++)
    {
        vAllIndices.push_back(i);
    }

    vector<vector<size_t>> sets = vector<vector<size_t>>(MAX_ITERATIONS, vector<size_t>(8, 0));

    srand(0);
    for(int i = 0; i < MAX_ITERATIONS; i++)
    {
        vAvailableIndices = vAllIndices;
        for(int j = 0; j < 8; j++)
        {
            int randNum = rand()%(vAvailableIndices.size() - 1);
            int idx = vAvailableIndices[randNum];

            sets[i][j] = idx;

            vAvailableIndices[randNum] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
    }

    return sets;
}

cv::Mat FindFundamental(vector<cv::KeyPoint>& points1, vector<cv::KeyPoint>& points2, vector<cv::DMatch>& good_matches, vector<vector<size_t>>& sets)
{
    const int N = points1.size();
    cv::Mat F21;
    vector<bool> vbInliers;
    
    assert(points1.size() == points2.size());

    cv::Mat T1, T2;
    vector<cv::Point2f> vPn1, vPn2;
    Normalize(points1, vPn1, T1);
    Normalize(points2, vPn2, T2);

    float score = 0.0;
    vector<bool> vbMatchesInliers = vector<bool>(N, false);

    vector<cv::Point2f> vPnl1i(8);
    vector<cv::Point2f> vPnl2i(8);
    cv::Mat F21i;
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    for(int i = 0; i<MAX_ITERATIONS; i++)
    {
        for(int j = 0; j<8; j++)
        {
            int idx = sets[i][j];
            vPnl1i[j] = vPn1[good_matches[idx].queryIdx];
            vPnl2i[j] = vPn2[good_matches[idx].trainIdx];
        }

        cv::Mat Fn = ComputeF21(vPnl1i, vPnl2i);

        F21i = T2.t()*Fn*T1;
        F21i = F21i/F21i.at<float>(2,2);

        currentScore = CheckFundamental(points1, points2, good_matches, F21i, vbCurrentInliers, 1.0);
        if(currentScore > score)
        {
            score = currentScore;
            F21 = F21i.clone();
            vbInliers = vbCurrentInliers;
        }
    }
    cout<<"score: "<<score<<endl;

    return F21;
}

cv::Mat ComputeF21(vector<cv::Point2f>& vPn1i, vector<cv::Point2f>& vPn2i)
{
    int N = vPn1i.size();
    cv::Mat A(N, 9, CV_32FC1);
    cv::Matx<float, 9, 9> A1;

    for(int i = 0; i<N; i++)
    {
        float u1 = vPn1i[i].x;
        float v1 = vPn1i[i].y;
        float u2 = vPn2i[i].x;
        float v2 = vPn2i[i].y;

        A.at<float>(i,0) = u2*u1;
        A.at<float>(i,1) = u2*v1;
        A.at<float>(i,2) = u2;
        A.at<float>(i,3) = v2*u1;
        A.at<float>(i,4) = v2*v1;
        A.at<float>(i,5) = v2;
        A.at<float>(i,6) = u1;
        A.at<float>(i,7) = v1;
        A.at<float>(i,8) = 1;

        cv::Vec<float, 9> r(u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, 1);
        A1 += r*r.t();
    }

    cv::Mat U, D, Vt;
    cv::SVD::compute(A, D, U, Vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    cv::Mat Fpre = Vt.row(8).reshape(0, 3);
    cv::SVD::compute(Fpre, D, U, Vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    D.at<float>(2,2) = 0;
    cv::Mat F21 = U*cv::Mat::diag(D)*Vt;

    // F21 = F21/F21.at<float>(2,2);

    return F21;
}

float CheckFundamental(vector<cv::KeyPoint>& points1, vector<cv::KeyPoint>& points2, vector<cv::DMatch>& good_matches, cv::Mat& F21, vector<bool>& vbCurrentInliers, float sigma)
{
    const int N = good_matches.size();

    const float f11 = F21.at<float>(0,0);
    const float f12 = F21.at<float>(0,1);
    const float f13 = F21.at<float>(0,2);
    const float f21 = F21.at<float>(1,0);
    const float f22 = F21.at<float>(1,1);
    const float f23 = F21.at<float>(1,2);
    const float f31 = F21.at<float>(2,0);
    const float f32 = F21.at<float>(2,1);
    const float f33 = F21.at<float>(2,2);

    float score = 0;
    const float th = 3.841;
    const float thScore = 5.991;
    const float invSigmaSquare = 1.0/(sigma*sigma);

    for(int i = 0; i<N; i++)
    {
        bool bIn = true;

        float u1 = points1[good_matches[i].queryIdx].pt.x;
        float v1 = points1[good_matches[i].queryIdx].pt.y;
        float u2 = points2[good_matches[i].trainIdx].pt.x;
        float v2 = points2[good_matches[i].trainIdx].pt.y;

        float a2 = f11*u1+f12*v1+f13;
        float b2 = f21*u1+f22*v1+f23;
        float c2 = f31*u1+f32*v1+f33;

        float num2 = (a2*u2+b2*v2+c2);
        float squareDist1 = num2*num2/(a2*a2+b2*b2);

        float chiSquare1 = squareDist1 / invSigmaSquare;
        if(chiSquare1 > th)
            bIn = false;
        else
            score += thScore - chiSquare1;

        float a1 = f11*u2+f12*v2+f13;
        float b1 = f21*u2+f22*v2+f23;
        float c1 = f31*u2+f32*v2+f33;

        float num1 = (a1*u1+b1*v1+c1);
        float squareDist2 = num1*num1/(a1*a1+b1*b1);

        float chiSquare2 = squareDist2 / invSigmaSquare;
        if(chiSquare2 > th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        if(bIn)
            vbCurrentInliers[i] = true;
        else
            vbCurrentInliers[i] = false;
    }

    return score;
}

void DecomposeE(cv::Mat& E, cv::Mat& R, cv::Mat& t, cv::Mat& K, vector<cv::KeyPoint>& keypoints_1, vector<cv::KeyPoint>& keypoints_2, vector<cv::DMatch>& good_matches, vector<cv::Point3f>& vP3D1)
{
    cv::Mat U,D,Vt;
    cv::SVD::compute(E, D, U, Vt, cv::SVD::FULL_UV);

    if(cv::determinant(U) < 0) U *= -1.0;
    if(cv::determinant(Vt) < 0) Vt *= -1.0;
    cv::Mat R1, R2, t1, t2;
    cv::Mat W(3,3,CV_32FC1,cv::Scalar(0));
    W.at<float>(0,1)=1.0;
    W.at<float>(1,0)=-1.0;
    W.at<float>(2,2)=1.0;
    W.convertTo(W, E.type());
    // cout<<U.type()<<endl;
    R1 = U*W*Vt;
    R2 = U*W.t()*Vt;
    t = U.col(2)*1.0;
    t = t/cv::norm(t);
    t1 = t;
    t2 = -t;
    cv::Mat R_gt, t_gt;
    GetTruthPose(R_gt, t_gt);
    t_gt = -t_gt;
    vector<cv::Point3f> vP3D11, vP3D12, vP3D13, vP3D14;

    int nGood1 = CheckRT(R1, t1, K, keypoints_1, keypoints_2, good_matches, vP3D11);
    int nGood2 = CheckRT(R1, t2, K, keypoints_1, keypoints_2, good_matches, vP3D12);
    int nGood3 = CheckRT(R2, t1, K, keypoints_1, keypoints_2, good_matches, vP3D13);
    int nGood4 = CheckRT(R2, t2, K, keypoints_1, keypoints_2, good_matches, vP3D14);
    // int nGood5 = CheckRT(R_gt, t_gt, K, keypoints_1, keypoints_2, good_matches);
    int nGood = min(nGood1, min(nGood2, min(nGood3, nGood4)));
    cout<<"1: "<<nGood1<<" 2: "<<nGood2<<" 3: "<<nGood3<<" 4: "<<nGood4<<endl;

    if(nGood == nGood1)
    {   
        R = R1;
        t = t1;
        vP3D1 = vP3D11;
    }
    else if(nGood == nGood2)
    {
        R = R1;
        t = t2;
        vP3D1 = vP3D12;
    }
    else if(nGood == nGood3)
    {
        R = R2;
        t = t1;
        vP3D1 = vP3D13;
    }
    else
    {
        R = R2;
        t = t2;
        vP3D1 = vP3D14;
    }
}

int CheckRT(cv::Mat& R, cv::Mat& t, cv::Mat& K, vector<cv::KeyPoint>& keypoints_1, vector<cv::KeyPoint>& keypoints_2, vector<cv::DMatch>& good_matches, vector<cv::Point3f>& vP3D1)
{
    vP3D1.resize(keypoints_1.size());
    cv::Mat P1(3,4,CV_32FC1,cv::Scalar(0));
    K.copyTo(P1.rowRange(0,3).colRange(0,3));
    // cout<<"P1: \n"<<P1<<endl;

    cv::Mat P2(3,4,CV_32FC1,cv::Scalar(0));
    R.copyTo(P2.rowRange(0,3).colRange(0,3));
    t.copyTo(P2.rowRange(0,3).col(3));
    P2 = K*P2;

    cout<<"Triangulate points : "<<good_matches.size()<<endl;

    cv::Mat O1 = cv::Mat::zeros(3,1,CV_32FC1);
    cv::Mat O2 = -R.t()*t;
    int cnt1 = 0, cnt2 = 0;
    for(int i = 0; i<good_matches.size(); i++)
    {
        cv::Point2f kp1 = keypoints_1[good_matches[i].queryIdx].pt;
        cv::Point2f kp2 = keypoints_2[good_matches[i].trainIdx].pt;
        cv::Mat p3dC1;
        Triangulate(kp1, kp2, P1, P2, p3dC1);
        if(p3dC1.at<float>(2) < 0)
            cnt1++;
        // cout<<"p3dC1: "<<p3dC1<<endl;

        cv::Mat p3dC2 = R*p3dC1 + t;
        if(p3dC2.at<float>(2) < 0)
            cnt2++;
            // cout<<"error 2 "<<i<<endl;

        cv::Mat normal1 = p3dC1 - O1;
        double dist1 = cv::norm(normal1);

        cv::Mat normal2 = p3dC1 - O2;
        double dist2 = cv::norm(normal2);

        double cosparallax = normal1.dot(normal2)/(dist1*dist2);
        // cout<<"cosparallax: "<<cosparallax<<endl;
        // cout<<"angle: "<<acos(cosparallax)*180/CV_PI<<endl;

        if(p3dC1.at<float>(2) > 0 && p3dC1.at<float>(2) > 0)
            vP3D1[good_matches[i].queryIdx] = cv::Point3f(p3dC1.at<float>(0), p3dC1.at<float>(1), p3dC1.at<float>(2));
    }

    return cnt1+cnt2;
}

void Triangulate(cv::Point2f& kp1, cv::Point2f& kp2, cv::Mat& P1, cv::Mat& P2, cv::Mat& p3dC1)
{
    float u1 = float(kp1.x);
    float v1 = float(kp1.y);
    float u2 = float(kp2.x);
    float v2 = float(kp2.y);

    cv::Mat A(4,4,CV_32FC1,cv::Scalar(0));

    A.row(0) = v1*P1.row(2) - P1.row(1);
    A.row(1) = u1*P1.row(2) - P1.row(0);
    A.row(2) = v2*P2.row(2) - P2.row(1);
    A.row(3) = u2*P2.row(2) - P2.row(0);

    cv::Mat U, W, Vt;
    cv::SVD::compute(A, W, U, Vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    p3dC1 = Vt.row(3).t();
    p3dC1 = p3dC1.rowRange(0,3) / p3dC1.at<float>(3);
    // cout<<"p3dC1: \n"<<p3dC1<<endl;
    // if(p3dC1.at<float>(2) < 0)
    //     cout<<"error!"<<endl;
}

void Normalize(vector<cv::KeyPoint>& vKeys, vector<cv::Point2f>& normalizedKeys, cv::Mat& T)
{
    float meanX = 0;
    float meanY = 0;
    const int N = vKeys.size();

    normalizedKeys.resize(N);

    for(int i = 0; i<N; i++)
    {
        meanX += vKeys[i].pt.x;
        meanY += vKeys[i].pt.y;
    }

    meanX /= N;
    meanY /= N;

    float meanDevX = 0;
    float meanDevY = 0;
    for(int i = 0; i<N; i++)
    {
        normalizedKeys[i].x = vKeys[i].pt.x - meanX;
        normalizedKeys[i].y = vKeys[i].pt.y - meanY;

        meanDevX += fabs(normalizedKeys[i].x);
        meanDevY += fabs(normalizedKeys[i].y);
    }

    float sX = N*1.0/meanDevX;
    float sY = N*1.0/meanDevY;

    for(int i = 0; i< N; i++)
    {
        normalizedKeys[i].x *= sX;
        normalizedKeys[i].y *= sY;
    }

    T = cv::Mat::eye(3,3,CV_32FC1);
    T.at<float>(0,0) = sX;
    T.at<float>(1,1) = sY;
    T.at<float>(0,2) = -meanX*sX;
    T.at<float>(1,2) = -meanY*sY;
}

void GetPredict(vector<cv::KeyPoint>& keypoints_1, vector<cv::KeyPoint>& keypoints_2, vector<cv::KeyPoint>& keypoints_2_hat,vector<cv::DMatch>& good_matches, cv::Mat& R, cv::Mat& t, vector<cv::Point3f>& vP3D1)
{
    cv::Mat p3dC1(3,1,CV_32FC1);
    cv::Mat p3dC2(3,1,CV_32FC1);

    float fx = 718.856;
    float cx = 607.1928;
    float fy = 718.856;
    float cy = 185.2157;

    for(int i = 0; i<good_matches.size(); i++)
    {
        cv::Point3f p1 = vP3D1[good_matches[i].queryIdx];
        p3dC1.at<float>(0) = p1.x;
        p3dC1.at<float>(1) = p1.y;
        p3dC1.at<float>(2) = p1.z;
        p3dC2 = R*p3dC1+t;
        p3dC2 = p3dC2 / p3dC2.at<float>(2);
        cv::KeyPoint p2_hat;
        p2_hat.pt.x = p3dC2.at<float>(0)*fx+cx;
        p2_hat.pt.y = p3dC2.at<float>(1)*fy+cy;

        keypoints_2_hat[good_matches[i].trainIdx] = p2_hat;
    }
}


// cv::Mat R, t1;
    // cv::recoverPose(E, points1, points2, K, R1, t);
    // Eigen::Matrix<float, 3, 3> E_eigen;
    // cv::cv2eigen(E, E_eigen); 
    // Eigen::JacobiSVD<Eigen::Matrix3d> svd(E_eigen, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // Eigen::Vector3d sigma1 = svd.singularValues();
    // Eigen::Matrix3d SIGMA;
    // cout<<"sigma: \n"<<sigma1<<endl;
    // SIGMA<<(sigma1(0,0)+sigma1(1,0))/2, 0, 0, 0, (sigma1(0,0)+sigma1(1,0))/2, 0, 0, 0, 0;
    // cout<<"SIGMA: \n"<<SIGMA<<endl;

    // Eigen::Matrix3d t_1;
    // Eigen::Matrix3d t_2;
    // Eigen::Matrix3d R1;
    // Eigen::Matrix3d R2;

    // Eigen::Matrix3d R_z1 = Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();
    // Eigen::Matrix3d R_z2 = Eigen::AngleAxisd(-M_PI/2, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();
    // cout<<"R_z1: \n"<<R_z1<<endl<<"R_z2: \n"<<R_z2<<endl;

    // R1 = svd.matrixU()*R_z1*svd.matrixV().transpose();
    // R1 = svd.matrixU()*R_z2*svd.matrixV().transpose();
    // t_1 = svd.matrixU()*R_z1*SIGMA*svd.matrixU().transpose();
    // t_1 = svd.matrixU()*R_z2*SIGMA*svd.matrixU().transpose();
    // cout<<"t1: \n"<<t_1<<endl;
    // cv::Mat t(3,1,CV_32FC1);
    // cv::eigen2cv(R1, R);
    // cv::eigen2cv(t_1, t1);
    // t.at<float>(2,0) = -t1.at<float>(0,1);
    // t.at<float>(1,0) = t1.at<float>(0,2);
    // t.at<float>(0,0) = -t1.at<float>(1,2);

    // cout<<"R: "<<R<<endl;
    // cout<<"t: "<<t<<endl;

//dataset KITTI