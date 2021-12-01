#include <cstdio>
#include <iostream>

using namespace std;

int arr[5000000];

int Divide(int a[], int start, int end)
{
    int x = a[start];
    while(start != end)
    {
        while(start < end && a[end] >= x)
            --end;
        if(start < end)
        {
            a[start] = a[end];
            ++start;
        }
        while(start < end && a[start] <= x)
            ++start;
        if(start < end)
        {
            a[end] = a[start];
            --end;
        }
    }
    a[start] = x;

    return start;   
}

int QuickSelect(int a[], int k, int start, int end)
{
    int div = Divide(a, start, end); // divide arr into >= a[0] and <= a[0]; div is the dividing index
    if(div == k)
        return a[div];
    else if(div > k)
        return QuickSelect(a, k, start, div-1); // k-th is in the left
    else
        return QuickSelect(a, k, div+1, end); // k-th is in the right
}

int main()
{
    int k = 0;
    int n = 0;
    int temp = 0;

    // input
    scanf("%d", &n);
    scanf("%d", &k);
    for(int i = 0; i < n; ++i)
    {
        scanf("%d", &temp);
        arr[i] = temp;
    }

    // finding k, which is actually k-1 because arrays start from 0.
    cout << QuickSelect(arr, k-1, 0, n-1);

    return 0;
}