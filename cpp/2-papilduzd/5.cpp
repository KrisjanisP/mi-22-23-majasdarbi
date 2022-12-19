#include "utils.hpp"

int main() {
    cout<<"5. UZDEVUMS\n";
    string virkne;
    cout<<"Ievadiet simbolu virkni: ";
    getline(cin, virkne);
    string result;
    for(char c: virkne)
        if(toupper(c)>='A'&&toupper(c)<='Z')
            result.push_back(c);
    cout<<result<<endl;
}