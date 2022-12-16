#include "utils.hpp"

int main() {
    cout<<"4. UZDEVUMS\n";
    int n;
    input("Ievadiet skaitli: ",n);
    int r = 0; // reversed n
    while(n>0) {
        r = r*10+n%10;
        n/=10;
    }
    cout<<"Apgrieztais skaitlis: "<<r<<endl;
}