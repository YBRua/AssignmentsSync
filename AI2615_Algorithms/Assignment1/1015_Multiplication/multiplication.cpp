#include <iostream>
#define ARRAY_SIZE 1000

using namespace std;


int a[ARRAY_SIZE];
int b[ARRAY_SIZE];
int lengthA = 0;
int lengthB = 0;

int output[ARRAY_SIZE];
int buffer[ARRAY_SIZE];

void ReadInput(int &length)
{
    char input;
    for(int i = 0; i < ARRAY_SIZE; ++i)
    {
        cin.get(input);
        if(input < '0' || input > '9')
            break;
        else
        {
            buffer[i] = input - '0';
            ++length;
        }
    }

    return;
}

void ReverseCopy(int from[], int to[], int length)
{
    for(int i = 0; i < length; ++i)
        to[length - 1 - i] = from[i];

    return;
}

void ClearBuffer()
{
    // Clear the buffer
    for(int i = 0; i < ARRAY_SIZE; ++i)
        buffer[i] = 0;

    return;
}

int HiAccAddition(int a[], int lengthA, int b[], int lengthB, int result[])
{
    int carry = 0;
    int sum = 0;
    int i = 0;
    // Regular bit-wise addition
    for(i = 0; i < ARRAY_SIZE; ++i)
    {
        if(i >= lengthA || i >= lengthB)
            break;
        else
        {
            sum = a[i] + b[i] + carry; // sum
            carry = 0; // reset carry
            if(sum >= 10)
            {
                sum -= 10;
                carry += 1;
            }
            result[i] = sum;
        }
    }
    if(lengthA > lengthB)
    {
        for(; i <= lengthA; ++i)
        {
            sum = a[i] + carry;
            carry = 0;
            if(sum >= 10)
            {
                sum -= 10;
                carry += 1;
            }
            result[i] = sum;
        }
    }
    else
    {
        for(; i <= lengthB; ++i)
        {
            sum = b[i] + carry;
            carry = 0;
            if(sum >= 10)
            {
                sum -= 10;
                carry += 1;
            }
            result[i] = sum;
        }
    }
    return result[i - 1] == 0 ? (i - 1) : i;
}

int HiAccMultiplication(int a[], int lengthA, int b[], int lengthB, int result[])
{
    int i = 0;
    int j = 0;
    int product = 0;
    int carry = 0;
    int offset = 0;
    int resultLength = 0;

    for(i = 0; i < lengthA; ++i)
    {
        ClearBuffer(); // Clear buffer. Buffer will be used to store the result of an iteration on j.

        for(j = 0; j < lengthB; ++j)
        {
            // Calculate bit-wise product.
            product = a[i] * b[j] + carry;
            carry = 0;
            while(product >= 10)
            {
                product -= 10;
                carry += 1;
            }
            buffer[j + offset] = product;
        }

        // Deal with the last carry.
        if(carry != 0)
        {
            buffer[j + offset] = carry;
            carry = 0;
        }
        else
        {
            j -= 1; // reset pointer when no carry is needed.
        }

        // Add result to the total product.
        resultLength = HiAccAddition(buffer, j + offset + 1, result, resultLength, result);
        // Apply bit offset.
        ++ offset;
    }
    return result[resultLength - 1] == 0 ? (resultLength - 1) : resultLength;
}

int main()
{
    ReadInput(lengthA);
    ReverseCopy(buffer, a, lengthA);

    ReadInput(lengthB);
    ReverseCopy(buffer, b, lengthB);

    if(a[lengthA - 1] == 0 || b[lengthB - 1] == 0)
    {
        cout << 0 << endl;
        return 0;
    }

    int outputLength = HiAccMultiplication(a, lengthA, b, lengthB, output);

    for(int i = 0; i < outputLength; ++i)
        cout << output[outputLength - 1 - i];
    cout << endl;

    return 0;
}