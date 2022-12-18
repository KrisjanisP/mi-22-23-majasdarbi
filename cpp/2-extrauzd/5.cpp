#include "utils.hpp"

string blue(string txt) {
    string res = "\033[1;34m";
    res += txt;
    res += "\033[0m";
    return res;
}

string yellow(string txt) {
    string res = "\033[1;33m";
    res += txt;
    res += "\033[0m";
    return res;
}

string red(string txt) {
    string res = "\033[1;31m";
    res += txt;
    res += "\033[0m";
    return res;
}

int main() {
    cout<<"5.1. - 5.3. UZDEVUMS\n";

    for(int i=1;i<=5;i++) {
        for(int j=1;j<=5;j++) {

            if(j%2) cout<<red(to_string(i));
            else cout<<i;

            cout<<" * ";

            if(j%2==0) cout<<blue(to_string(j));
            else cout<<j;

            cout<<" = ";

            cout<<yellow(to_string(i*j))<<endl;
        }
    }
}