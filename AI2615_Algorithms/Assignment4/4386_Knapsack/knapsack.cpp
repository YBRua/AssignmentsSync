// Claim: dynamic programming is totally difficult.

#include <iostream>

using namespace std;

int weight[301];
int value[301];
// subproblem: largest value we get using the first i items with a bag of size j.
int knapsack[2][100001];

int main()
{
    int n; // num of items
    int m; // size of bag

    cin >> n >> m;

    for (int i = 1; i <= n; ++i)
    {
        cin >> weight[i];
        cin >> value[i];
    }

    for (int j = 0; j <= m; ++j)
        knapsack[0][j] = 0;

    for (int i = 1; i <= n; ++i)
    {
        for (int j = 1; j <= m; ++j)
        {
            if (weight[i] > j)
                knapsack[1][j] = knapsack[0][j];
            else
            {
                int donttake = knapsack[0][j];
                int take = knapsack[0][j-weight[i]] + value[i];
                knapsack[1][j] = take > donttake ? take : donttake;
            }
        }
        for (int j = 1; j <= m; ++j)
            knapsack[0][j] = knapsack[1][j];
    }

    cout << knapsack[0][m];

}
