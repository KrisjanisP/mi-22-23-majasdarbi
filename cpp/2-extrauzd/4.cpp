#include "utils.hpp"

bool is_palindrome(string str) {
    for(int i=0;i<str.size()/2;i++){
        if(str[i]!=str[str.size()-1-i])
            return false;
    }
    return true;
}

int main() {
    cout<<"4.1. - 4.3. UZDEVUMS\n";

    cout<<"4.1. UZDEVUMS\n";
    int number;
    input("Ievadiet skaitli: ",number);
    if(is_palindrome(to_string(number)))
        cout<<"Ievadītais skaitlis IR palindroms!\n";
    else
        cout<<"Ievadītais skaitlis NAV palindroms!\n";

    cout<<"4.2. UZDEVUMS\n";
    input("Ievadiet skaitli: ", number);

    int x = 1; // power of three
    while(x<number) x*=3;

    int lst = 0; // last digit of number converted to base 3
    while(x>0) {
        lst = 0;
        while(number>=x)
            number -= x, lst++;
        x/=3;
    }

    if(lst) cout<<"Ievadītais skaitlis NEdalās ar 3\n";
    else cout<<"Ievadītais skaitlis DALĀS ar 3\n";

    cout<<"4.3. UZDEVUMS\n";
    input("Ievadiet skaitli: ", number);
    string number_str;
    for(int i=0;i<5;i++)
        number_str += to_string(number);
    string result;
    int remainder = 0;
    for(int i=0;i<number_str.size();i++) {
        remainder *= 10;
        remainder += number_str[i]-'0';
        if(!i&&remainder<2) continue; // don't add the leading zero
        result += to_string(remainder/2);
        remainder %= 2;
    }
    cout<<result;
    if(remainder) cout<<".5"<<endl;
    else cout<<endl;
}