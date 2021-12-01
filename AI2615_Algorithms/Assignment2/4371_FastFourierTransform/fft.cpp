#include <iostream>
#include <complex>
#include <cmath>

#define LENGTH 524288
#define PI 3.1415926535897932384626

using namespace std;

complex<double> p[LENGTH] = {0};
complex<double> q[LENGTH] = {0};
complex<double> ans[LENGTH] = {0};
complex<double> buffer[LENGTH] = {0};

/*
class Complex
{
public:
    double real;
    double image;
};
*/

void FFT(complex<double> arr[], int length, complex<double> w)
{
    if(length == 1)
        return;
    
    // backup all data
    for(int i = 0; i < length; ++i)
    {
        buffer[i] = arr[i];
    }

    // separate even and odd terms
    for(int i = 0; i < length/2; ++i)
    {
        arr[i] = buffer[2*i];
        arr[length/2 + i] = buffer[2*i + 1];
    }

    // even numbers
    FFT(arr, length/2, w*w);
    // odd numbers
    FFT(arr+length/2, length/2, w*w);

    complex<double> temp = 1;
    for(int i = 0; i < length/2; ++i)
    {
        buffer[i] = arr[i] + temp * arr[i + length/2];
        buffer[i + length/2] = arr[i] - temp * arr[i + length/2];
        temp *= w;
    }
    for(int i = 0; i < length; ++i)
    {
        arr[i] = buffer[i];
    }

    return;
}

void MultByFFT(complex<double> p[], complex<double> q[], int length)
{
    complex<double> useMeaningfulVariableName(cos(2*PI/length), sin(2*PI/length));

    FFT(p, length, useMeaningfulVariableName);
    FFT(q, length, useMeaningfulVariableName);

    for(int i = 0; i < length; ++i)
    {
        ans[i] = p[i] * q[i];
    }

    FFT(ans, length, conj(useMeaningfulVariableName));

    return;
}

int main()
{
    int m;
    int n;
    int inputCoefficient;

    // input
    cin >> m >> n;

    for(int i = 0; i <= m; ++i)
    {
        scanf("%d", &inputCoefficient);
        p[i] = complex<double>(inputCoefficient, 0);
    }
    for(int i = 0; i <= n; ++i)
    {
        scanf("%d", &inputCoefficient);
        q[i] = complex<double>(inputCoefficient, 0);
    }

    // Fast Fourier Toileting
    MultByFFT(p, q, LENGTH);

    // Output
    for(int i = 0; i < m + n + 1; ++i)
        printf("%d ", static_cast<int>((ans[i].real()/LENGTH) + 0.5));

    return 0;
}