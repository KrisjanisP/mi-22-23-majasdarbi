#include <bits/stdc++.h>

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
    cin.ignore(69420, '\n'); // clear buffer
}


template <typename T, typename Lambda>
void input(const string& prompt, T &variable, Lambda correct)
{
    cout << prompt;
    cin >> variable;
    if (cin.fail() || !correct(variable))
    {
        cin.clear();
        cin.ignore(69420, '\n');
        std::cout << "Ievades kļūda!\n";
        return input(prompt, variable, correct);
    }
    cin.ignore(69420, '\n'); // clear buffer
}

void inputLine(const string& prompt, string &variable) {
    cout<<prompt;
    getline(cin, variable);
    if(cin.fail()) {
        cin.clear();
        cin.ignore(69420, '\n');
        cout<<"Ievades kļūda!\n";
        return input(prompt,variable);
    }
}

void clearConsole()
{
    #ifdef _WIN32
    system("cls");      // clear console on windows
    #else
    system("clear");    // clear console on linux
    #endif
}

void pauseConsole()
{
    cout << "Nospiediet enter, lai turpināt!" << endl;
    cin.ignore(69420, '\n'); // clear input buffer
}
