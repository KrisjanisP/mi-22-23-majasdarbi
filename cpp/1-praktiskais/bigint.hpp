// Author: Krišjānis Petručeņa
#include <string>
class BigInt {
private:
    std::string data;
public:
    BigInt();
    BigInt(long long x);
    BigInt& operator+=(const BigInt& rhs); // not implemented yet
    BigInt& operator*=(long long rhs);
    std::string toString();
};