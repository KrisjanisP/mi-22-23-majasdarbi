// Author: Krišjānis Petručeņa
#include <vector>
#include <cstring>
#include <string>
#include <istream>
#include "string.hpp"
#include <iostream>
#ifdef _WIN32
#include "windows.h"
#endif

kpetrucena::string::string(){}

kpetrucena::string::string(const char str[]){
    push_back(str);
}

void kpetrucena::string::push_back(const char str[]) {
    int left = 0;
    for(int i=0;str[i]!='\0';i++){
        if(left==0) {
            int ones = 0; // leading ones in binary
            int x = 128;
            while(str[i]&x) {
                ones++;
                x>>=1;
            }
            if(ones==0) left = 1;
            else left = ones;
            data.push_back(new char[5]);
            // set last element of data to '\0'
            memset(data.back(),0,sizeof(data.back()));
            length.push_back(0);
        }
        left--;
        data.back()[length.back()]=str[i];
        length.back()++;
    }
}

kpetrucena::string::~string() {
    // vector elements get automatically destructed
}

std::string kpetrucena::string::toSTDString() const {
    std::string res;
    for(int i=0;i<data.size();i++){
        for(int j=0;j<length[i];j++)
            res += data[i][j];
    }
    return res;
}

int kpetrucena::string::size() const {
    return data.size();
}

char* kpetrucena::string::operator[](int i) {
    return data[i];
}

std::istream& operator>>(std::istream& is, kpetrucena::string& str)
{
    #ifdef _WIN32
    str = kpetrucena::string();
    while(true) {
        wchar_t wchar_answer[2];
        memset(wchar_answer,0,sizeof(wchar_answer));
        DWORD read_wchars = 0;
        ReadConsoleW(GetStdHandle(STD_INPUT_HANDLE),wchar_answer,1,&read_wchars, NULL);
        char char_answer[5];
        memset(char_answer,0,sizeof(char_answer));
        WideCharToMultiByte(CP_UTF8,0,wchar_answer,1,char_answer,5,NULL,NULL);
        if(char_answer[0]>=0&&char_answer[0]<33) break; // special symbols in ASCII table
        str.push_back(char_answer);
    }
    return is;
    #else
    std::string tmp;
    is>>tmp;
    str = kpetrucena::string(tmp.data());
    return is;
    #endif
}

std::ostream& operator<<(std::ostream& os, const kpetrucena::string& str)
{
    os << str.toSTDString();
    return os;
}

std::istream& getline(std::istream& is, kpetrucena::string& str) {
    #ifdef _WIN32
    str = kpetrucena::string();
    while(true) {
        wchar_t wchar_answer[2];
        memset(wchar_answer,0,sizeof(wchar_answer));
        DWORD read_wchars = 0;
        ReadConsoleW(GetStdHandle(STD_INPUT_HANDLE),wchar_answer,1,&read_wchars, NULL);
        char char_answer[5];
        memset(char_answer,0,sizeof(char_answer));
        WideCharToMultiByte(CP_UTF8,0,wchar_answer,1,char_answer,5,NULL,NULL);
        if(char_answer[0]>=0&&char_answer[0]<32) break; // special symbols in ASCII table
        str.push_back(char_answer);
    }
    return is;
    #else
    std::string tmp;
    getline(is,tmp);
    str = kpetrucena::string(tmp.data());
    return is;
    #endif
}