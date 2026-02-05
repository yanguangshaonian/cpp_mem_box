#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include "box.hpp"

using namespace std;

template<class T>
class alignas(64) MyVec {
    uint64_t VEC_MAGIC = 0x20260205;
    private:
    public:
        uint64_t magic;
        uint64_t current_size;
        uint64_t capacity;
        T data[];

        
};


int main(const int argc, char* argv[]) {

    cout << "hello" << endl;
}