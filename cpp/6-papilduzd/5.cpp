#include "utils.hpp"

int *minElement(int *arr, int arrSize)
{
    int *res = &arr[0];
    for (int *i = new int(0); (*i) < arrSize; (*i)++)
        if (arr[*i] < (*res))
            res = &arr[*i];
    return res;
}

int main()
{
    cout << "5. UZDEVUMS" << endl;
    int arr[10] = {6, 4, 6, 8, 10, 12, 14, 16, 18, 20};
    cout << "Mazākais elements: " << *minElement(arr, 10) << endl;
}