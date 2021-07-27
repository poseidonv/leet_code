#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <ctime>
#include <iomanip>
#include <algorithm>
#include <list>
#include <queue>
#include <set>
#include <map>
#include <stack>
#include <cstring>
#include <sstream>
#include <typeinfo>
#include <numeric>
#include "omp.h"
#include <opencv2/core/core.hpp>

using namespace std;

template <typename T>
void PrintVector(vector<T> vecs)
{
    typename vector<T>::iterator it = vecs.begin();
    while (it != vecs.end())
    {
        // cout<<it->first<<' '<<it->second<<endl;
        cout << *it << endl;
        it++;
    }
}

template <>
void PrintVector(vector<pair<int, int>> vecs)
{
    vector<pair<int, int>>::iterator it = vecs.begin();
    while (it != vecs.end())
    {
        cout << it->first << ' ' << it->second << endl;
        it++;
    }
}

// class Solution {
// public:
//     int numSubarraysWithSum(vector<int>& nums, int goal) {

//     }
// };

// struct comp
// {
//     bool operator()(pair<int, int> a, pair<int, int> b)
//     {
//         return a.first < b.first || a.first == b.first && a.second < b.second;
//     }
// };

ostream &operator<<(ostream &cout, pair<int, int> &pair)
{
    cout<<pair.first<< " | "<<pair.second;
    return cout;
}

struct comp
{
    bool operator()(pair<int, pair<int,int>> a, pair<int, pair<int,int>> b)
    {
        return a.first > b.first;
    }
};

class Solution
{
public:
    int minPushBox(vector<vector<char>> &grid)
    {
        pair<int, int> T_coordinate = {0, 0};
        pair<int, int> B_coordinate = {0, 0};
        mvvbVisited = vector<vector<char>>(grid.size(), vector<char>(grid[0].size(), false));

        for(size_t i = 0; i<grid.size(); i++)
        {
            for(size_t j = 0; j<grid[i].size(); j++)
            {
                if(grid[i][j] == 'T')
                {
                    T_coordinate = {i, j};
                    grid[i][j] = '.';
                }
                else if(grid[i][j] == 'B')
                {
                    B_coordinate = {i, j};
                    grid[i][j] = '.';
                }
                else if (grid[i][j] == 'S')
                {
                    grid[i][j] = '.';
                }
            }
        }
        int distance = ( (T_coordinate.first - B_coordinate.first)*(T_coordinate.first - B_coordinate.first) + (T_coordinate.second - B_coordinate.second)*(T_coordinate.second - B_coordinate.second) ) + 0;
        mpque.push({distance, {B_coordinate.first, B_coordinate.second}});
        while(!mpque.empty())
        {
            if(mpque.top().second.first == T_coordinate.first && mpque.top().second.second == T_coordinate.second)
                break;
            FindMinimialDistance(grid, T_coordinate, mpque.top().second , 0);
        }
        cout << mpque.top().first<<endl;
        // co
        return -1;
    }

private:
    int FindMinimialDistance(vector<vector<char>> &grid, pair<int, int> T, pair<int, int> B, int previousDistance)
    {
        int x = B.first;
        int y = B.second;
        
        mpque.pop();
        mvvbVisited[x][y] = true;

        int distance1 = INT_MAX;
        int distance2 = INT_MAX;
        int distance3 = INT_MAX;
        int distance4 = INT_MAX;
        if((x-1) > 0 && (x+1) < grid.size())
        {
            if(grid[x-1][y] == '.' && grid[x-1][y])
            {
                distance1 = ( (T.first - (x-1))*(T.first - (x-1)) + (T.second - y)*(T.second - y) ) + (previousDistance + 1);
                if(distance1 < INT_MAX && !mvvbVisited[x-1][y])
                    mpque.push({distance1, {x-1, y}});
                distance2 = ( (T.first - (x+1))*(T.first - (x+1)) + (T.second - y)*(T.second - y) ) + (previousDistance + 1);
                if(distance2 < INT_MAX && !mvvbVisited[x+1][y])
                    mpque.push({distance2, {x+1, y}});
            }
        }
        if((y-1) > 0 && (y+1) < grid[x].size())
        {
            if(grid[x][y-1] == '.' && grid[x][y+1] == '.')
            {
                distance3 = ( (T.first - x)*(T.first - x) + (T.second - (y-1))*(T.second - (y-1)) ) + (previousDistance + 1);
                if(distance3 < INT_MAX && !mvvbVisited[x][y-1])
                    mpque.push({distance3, {x, y-1}});
                distance4 = ( (T.first - x)*(T.first - x) + (T.second - (y+1))*(T.second - (y+1)) ) + (previousDistance + 1);
                if(distance4 < INT_MAX && !mvvbVisited[x][y+1])
                    mpque.push({distance4, {x, y+1}});
            }
        }
            
        cout<<distance1 << " "<<distance2 << " " <<distance3 << " "<<distance4<<endl;
    }

private:
    priority_queue<pair<int, pair<int,int> >, vector<pair<int, pair<int, int> > >, comp> mpque;
    vector<vector<char>> mvvbVisited;
};

int main(int argc, char **argv)
{
    Solution solution;
    clock_t start, end;
    start = clock();
    vector<vector<char>> grid = {
        {'#', '#', '#', '#', '#', '#'},
        {'#', 'T', '#', '#', '#', '#'},
        {'#', '.', '.', 'B', '.', '#'},
        {'#', '.', '#', '#', '.', '#'},
        {'#', '.', '.', '.', 'S', '#'},
        {'#', '#', '#', '#', '#', '#'}};
    auto result = solution.minPushBox(grid);
    cout << "result: " << result << endl;
    // for(auto i : result)
    // {
    //     cout<<i<<endl;
    // }
    end = clock();
    double totalTime = (double)(end - start) / CLOCKS_PER_SEC;
    cout << "total time: " << totalTime * 1000 << "ms" << endl;

    return 0;
}
