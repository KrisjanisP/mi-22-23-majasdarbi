#include <iostream>

using namespace std;

template<typename T>
void input(const string& prompt, T &variable) {
    cout<<prompt;
    cin>>variable;
    if(cin.fail()) {
        cin.clear();
        cin.ignore(69420, '\n');
        cout<<"Ievades kļūda!\n";
        return input(prompt,variable);
    }
}