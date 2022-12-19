#include "utils.hpp"

int main() {
    cout<<"2. UZDEVUMS\n";
    float a[] = {1.8,2.2,3.4,4.1,1.7,2.6,3.5,4.1,1.0,2.9};
    float mx = a[0];
    for(float x: a) mx = max(mx, x);
    cout<<mx<<endl;
}