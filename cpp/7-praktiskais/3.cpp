#include "utils.hpp"

struct Data
{
    int num;
    char chr;
};

void printByValue(Data obj)
{
    cout << "Num: " << obj.num << ", Chr: " << obj.chr << endl;
}

void printByPointer(Data *ptrObj)
{
    cout << "Num: " << ptrObj->num << ", Chr: " << ptrObj->chr << endl;
}

void printByReference(Data &obj)
{
    cout << "Num: " << obj.num << ", Chr: " << obj.chr << endl;
}

int main()
{
    cout << "3. UZDEVUMS" << endl;

    Data data = {3, 'a'};

    printByValue(data);
    printByPointer(&data);
    printByReference(data);
}