#include <iostream>
#define ARRAY_SIZE 128
#define MOD 1000000007

using namespace std;

class Mat2
{
public:
    long long int arr[2][2];

    Mat2(long long int a, long long int b, long long int c, long long int d)
    {
        arr[0][0] = a;
        arr[0][1] = b;
        arr[1][0] = c;
        arr[1][1] = d;
    }

    void Output()
    {
        cout << arr[0][1];
    }

    void DebugOutput()
    {
        for(int i = 0; i < 2; ++i)
        {
            for(int j = 0; j < 2; ++j)
            {
                cout << arr[i][j] << ' ';
            }
            cout << endl;
        }
    }
};

Mat2 MatMul(Mat2 a, Mat2 b, int n)
{
    Mat2 result = Mat2(0,0,0,0);
    for(int i = 0; i < n; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            for(int k = 0; k < n; ++k)
            {
                result.arr[i][j] += (a.arr[i][k] * b.arr[k][j]) % MOD;
                result.arr[i][j] %= MOD;
            }
        }
    }

    return result;
}

int main()
{
    long long int n;
    int doMultiplication;
    Mat2 m = Mat2(0,1,1,1);
    Mat2 result = Mat2(0,1,1,1);

    cin >> n;

    while(n > 0)
    {
        doMultiplication = n % 2;
        if(doMultiplication)
        {
            result = MatMul(result, m, 2);
        }
        n /= 2;
        m = MatMul(m, m, 2);
    }

    result.Output();
    return 0;
}