#include "utils.hpp"

int main() {
    cout<<"1. UZDEVUMS\n";
    int a[] = {1,2,3,4,1,2,3,4,1,2};
    int sum = 0;
    for(int x: a) sum += x;
    cout<<(double)sum/(sizeof(a)/sizeof(a[0]))<<endl;
}