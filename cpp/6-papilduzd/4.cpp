#include "utils.hpp"

void printReverse(int *arr)
{
    for (int *i = new int(9); (*i) >= 0; (*i)--)
    {
        cout << arr[*i] << " ";
    }

    cout << endl;
}

int main()
{
    cout << "4. UZDEVUMS" << endl;
    int arr[10] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
    printReverse(arr);
}