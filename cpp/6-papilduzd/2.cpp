#include "utils.hpp"

void find(int *arr, int *searching)
{
    bool *found = new bool(false);
    for (int *i = new int(0); (*i) < 10; (*i)++)
    {
        if (arr[*i] == *searching)
            (*found) = true;
    }

    if (*found)
        cout << "Elements atrodas masīvā." << endl;
    else
        cout << "Elements neatrodas masīvā." << endl;
}

int main()
{
    cout << "2. UZDEVUMS" << endl;
    int arr[10] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
    int *searching = new int;
    input("Ievadiet meklējamo elementu: ", *searching);
    find(arr, searching);
}