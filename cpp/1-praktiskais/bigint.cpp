// Author: Krišjānis Petručeņa
#include "bigint.hpp"
#include <vector>

BigInt::BigInt() {
    data = "0";
}

BigInt::BigInt(long long x) {
    data = std::to_string(x);
}

std::string BigInt::toString() {
    return data;
}

// BigInt& BigInt::operator+=(const BigInt& rhs) {}

 
// multiplies num1 and num2, and prints result.
std::string multiply(std::string str1, std::string str2)
{
    int len1 = str1.size();
    int len2 = str2.size();
    if (len1+len2 == 0) return "0";

    std::vector<int> result(len1 + len2, 0);
 
    int i_n1 = 0, i_n2 = 0;
     
    for (int i=len1-1; i>=0; i--)
    {
        int carry = 0;
        int n1 = str1[i] - '0';
        i_n2 = 0;
                
        for (int j=len2-1; j>=0; j--)
        {
            int n2 = str2[j] - '0';
            int sum = n1*n2 + result[i_n1 + i_n2] + carry;
 
            carry = sum/10;
 
            result[i_n1 + i_n2] = sum % 10;
 
            i_n2++;
        }
 
        if (carry > 0)
            result[i_n1 + i_n2] += carry;
        i_n1++;
    }
 
    int i = result.size() - 1;
    while (i>=0 && result[i] == 0)
    i--;

    if (i == -1) return "0";
 
    std::string s = "";
    while (i >= 0)
        s += std::to_string(result[i--]);
 
    return s;
}

BigInt& BigInt::operator*=(long long x) {
    this->data = multiply(this->data, std::to_string(x));
    return *this;
}