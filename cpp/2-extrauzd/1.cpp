#include "utils.hpp"

bool is_prime(int x) {
    if(x<=1) return false;
    for(int i=2;i*i<=x;i++)
        if(x%i==0) return false;
    return true;
}

int main() {
    cout<<"1.1. - 1.3. UZDEVUMS\n";

    cout<<"1.1. UZDEVUMS\n";
    int n;
    input("Ievadiet n: ", n);
    int p = 2; // meklējamais pirmskailis
    // pārvietos p uz priekšu n-1 reizes
    // piestājot pie pirmskaitļiem
    for(int i=0;i<n-1;i++){
        p++;
        while(is_prime(p)==false)
            p++;
    }
    cout<<"n-tais pirmskaitlis: "<<p<<endl;

    cout<<"1.2. UZDEVUMS\n";
    input("Ievadiet n: ", n);
    bool found = false;
    for(int i=2;i<=n;i++) {
        int j = n-i;
        if(j<i) break;
        if(is_prime(i)&&is_prime(j)) {
            cout<<i<<" "<<j<<endl;
            found = true;
        }
    }
    if(!found) cout<<0<<endl;

    cout<<"1.3. UZDEVUMS\n";
    input("Ievadiet n: ", n);
    found = false;
    for(int i=2;i*i<=n;i++) {
        if(n%i==0&&is_prime(i)&&is_prime(n/i)) {
            cout<<i<<" "<<n/i<<endl;
            found = true;
        }
    }
    if(!found) cout<<0<<endl;
}