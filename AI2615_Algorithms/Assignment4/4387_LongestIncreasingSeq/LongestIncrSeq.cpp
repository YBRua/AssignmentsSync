#include <iostream>
#define SIZE 10010

using namespace std;


int arr[SIZE];
int previous[SIZE];
int length[SIZE];
 
void recursivePrint(int start)
{
    if (previous[start] != -1)
        recursivePrint(previous[start]);
    cout << arr[start] << ' ';
}
 
int main()
{
    int n; // length of sequence
 
    // initialize
    cin >> n;
    for (int i = 0; i < n; ++i)
    {
        cin >> arr[i];
        previous[i] = -1;
        length[i] = 1;
    }
    
    // dp
    for (int i = 1; i < n; ++i)
    {
        int maxLength = 0;
        int argmax = -1;
        // find max F(j)
        for (int j = 0; j < i; ++j)
        {
            if (arr[j] < arr[i] && length[j] > maxLength)
            {
                maxLength = length[j];
                argmax = j;
            }
        }
        // find the smallest among all max F(j)
        int minValue = 10000;
        for (int j = 0; j < i; ++j)
        {
            if (length[j] == maxLength)
            {
                if (minValue >= arr[j])
                {
                    minValue = arr[j];
                    argmax = j;
                }
            }
        }
        length[i] = maxLength + 1;
        previous[i] = argmax;
    }
 
    int argmax = -1;
    int maxLength = -1;
    for (int i = 0; i < n; ++i)
    {
        if (length[i] >= maxLength)
        {
            maxLength = length[i];
            argmax = i;
        }
    }
 
    cout << length[argmax] << endl;
    recursivePrint(argmax);
 
    return 0;
}