#include "utils.hpp"

int main() {
    cout<<"4. UZDEVUMS\n";
    string virkne;
    input("Ievadiet simbolu virkni: ", virkne);
    string result;
    int res = 0;
    for(char c: virkne) if(c=='a') res++;
    cout<<"\'a\' skaits: "<<res<<endl;
}