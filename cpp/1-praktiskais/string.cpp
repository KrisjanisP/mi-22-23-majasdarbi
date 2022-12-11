#include <vector>
#include <cstring>
#include <string>
#include <istream>
#include "string.h"

kpetrucena::string::string(){}

kpetrucena::string::string(const char str[]){
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
    std::string tmp;
    is>>tmp;
    str = kpetrucena::string(tmp.data());
    return is;
}

std::ostream& operator<<(std::ostream& os, const kpetrucena::string& str)
{
    os << str.toSTDString();
    return os;
}

std::istream& getline(std::istream& is, kpetrucena::string& str) {
    char ch;
    std::string tmp;
    while(is.get(ch)&&ch!='\n')
        tmp.push_back(ch);
    str = kpetrucena::string(tmp.data());
    return is;
}