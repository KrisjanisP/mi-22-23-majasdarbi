#include "utils.hpp"

bool is_prime(int x)
{
    if (x <= 1)
        return false;
    for (int i = 2; i * i <= x; i++)
    {
        if (x % i == 0)
            return false;
    }
    return true;
}

int main()
{
    cout << "2. UZDEVUMS" << endl;
    int arr[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    ofstream even("file1.txt"), odd("file2.txt"), prime("file3.txt");
    if (even.is_open() && odd.is_open() && prime.is_open())
    {
        for (int i = 0; i < 10; i++)
        {
            if (arr[i] % 2)
                odd << arr[i] << " ";
            else
                even << arr[i] << " ";
            if (is_prime(arr[i]))
                prime << arr[i] << " ";
        }
        even.close();
        odd.close();
        prime.close();
    }
    else
    {
        cout << "Neizdevās atvērt trīs nepieciešamos failus.\n";
    }
}