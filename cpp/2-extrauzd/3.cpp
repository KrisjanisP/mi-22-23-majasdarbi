#include "utils.hpp"
#include <algorithm>

int main() {
    cout<<"3.1. - 3.3. UZDEVUMS\n";

    cout<<"3.1. UZDEVUMS\n";
    int n;
    input("Ievadiet n: ", n);
    int sum = 0;
    while(n) {
        sum += (n%10)*(n%10);
        n/=10;
    }
    cout<<"ciparu kvadrātu summa: "<<sum<<endl;

    cout<<"3.2. UZDEVUMS\n";

    do {
        input("Ievadiet 4 ciparu skaitli: ", n);
        if(n<1000||n>9999) {
            cout<<"Ievadītais skaitlis nav 4 ciparus garš!\n";
        }
    }while(n<1000||n>=10000);

    int a[4] = {n%10, (n/10)%10, (n/100)%10, (n/1000)%10};

    for(int i=0;i<4;i++) { // i - index of the digit that we are not picking
        sort(a,a+4);
        swap(a[i], a[3]); // place digit out of sight
        sort(a,a+3);
        do {
            for(int i=0;i<3;i++)
                cout<<a[i];
            cout<<endl;
        }while(next_permutation(a,a+3));
    }

    cout<<"3.3. UZDEVUMS\n";
    
    input("Ievadiet veselu skaitli: ",n);
    int lst = n%10;
    while(n>=10) n/=10;
    int fst = n%10;
    cout<<"pirmā un pēdējā cipara summa: "<<fst+lst<<endl;

}