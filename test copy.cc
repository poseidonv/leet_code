#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <ctime>
#include <iomanip>

using namespace std;

class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        double median = 0;
        median = FindTwoMedian(nums1, 0, nums1.size(), nums2, 0, nums2.size());
        return median;
    }

    int FindTwoMedian(vector<int>& nums1, int low1, int high1, vector<int>& nums2, int low2, int high2)
    {
        if(low1 < high1 && low2 < high2)
        {
            int median1 = nums1[(low1+high1) / 2];
            int median2 = nums2[(low2+high2) / 2];

            if(median1 > median2)
            {
                FindTwoMedian(nums1, low1, (low1+high1) / 2, nums2, (low2+high2) / 2, high2);
            }
            else
            {
                FindTwoMedian(nums1, (low1+high1) / 2, high1, nums2, low2, (low2+high2) / 2);
            }
            // cout<<"high1: "<<high1<<" low1: "<<low1<<endl;
        }
        else
        {
            if(low1 < high1)
                return nums2[(low2+high2) / 2];
            else
                return nums1[(low1+high1) / 2];
        }
    }
};

int main(int argc, char** argv)
{
    Solution solution;
    clock_t start, end;

    start = clock();
    int a[2] = {1,3};
    int b[2] = {2,4};
    vector<int> nums1(a, a+2);
    vector<int> nums2(b, b+2);
    double result = solution.findMedianSortedArrays(nums1, nums2);
    cout<<"result: "<<result<<endl;
    end = clock();
    double totalTime = (double)(end - start) / CLOCKS_PER_SEC;
    cout<<"total time: "<<totalTime*1000<<"ms"<<endl;
    
    return 0;
}