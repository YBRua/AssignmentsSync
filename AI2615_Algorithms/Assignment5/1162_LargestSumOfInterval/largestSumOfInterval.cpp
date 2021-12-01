#include <iostream>

using namespace std;

int arr[200005];
int largestSum[200005];

inline int Max(int a, int b)
{
    return a > b ? a : b;
}

int main()
{
    int n;

    cin >> n;
    for (int i = 0; i < n; ++i)
    {
        cin >> arr[i];
    }

    largestSum[0] = arr[0];
    for (int i = 1; i < n; ++i)
    {
        largestSum[i] = Max(arr[i], arr[i] + largestSum[i - 1]);
    }

    int largest = -114514;
    for (int i = 0; i < n; ++i)
    {
        largest = largest > largestSum[i] ? largest : largestSum[i];
    }

    cout << largest;

    return 0;
}