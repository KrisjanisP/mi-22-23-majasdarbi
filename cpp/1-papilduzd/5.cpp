#include "utils.hpp"
#include <iomanip>

using namespace std;

int main() {
    int n, range;
    input("Ievadiet n: ", n);
    input("Ievadiet range: ", range);
    for(int i=1;i<=range;i++)
        cout<<n,
        cout<<setw(3)<<"*",
        cout<<setw(3)<<i,
        cout<<setw(3)<<"=",
        cout<<setw(3)<<n*i<<endl;
}