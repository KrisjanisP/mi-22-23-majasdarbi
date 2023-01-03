#pragma once
#include <iostream>
#include <vector>
#include <sstream>
#include <iostream>
#include <queue>
#include <stack>
#include <list>

using namespace std;

template <typename T>
string to_string(vector<T> vec)
{
    stringstream ss;
    for (int i = 0; i < vec.size(); i++)
    {
        ss << vec[i];
        if (i < vec.size() - 1)
            ss << ",";
    }
    return "{" + ss.str() + "}";
}

template <typename T, typename Lambda>
void input(const string &prompt, T &variable, Lambda correct)
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