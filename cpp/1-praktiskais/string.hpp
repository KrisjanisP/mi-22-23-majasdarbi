// Author: Krišjānis Petručeņa
#include <vector>
#include <cstring>
#include <string>

namespace kpetrucena {
    // my own string implementation
    // saved as vector of char arrays
    class string {
    private:
        std::vector<char*> data;
        std::vector<int> length;
    public:
        string();
        string(const char str[]);
        ~string();
        std::string toSTDString() const;
        int size() const;
        char* operator[](int i);
        void push_back(const char c[]);
    };
}

std::istream& operator>>(std::istream& is, kpetrucena::string& str);
std::ostream& operator<<(std::ostream& os, const kpetrucena::string& str);

std::istream& getline(std::istream& is, kpetrucena::string& str);