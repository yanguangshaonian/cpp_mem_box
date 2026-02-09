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

class MyObj : public ipc::shm::Boxed<MyObj> {
    public:
        uint64_t age1;
        uint64_t age;
        char name[40];
};

int main(const int argc, char* argv[]) {

    auto my_obj = ipc::shm::Box<MyObj>{};
    my_obj.attach_or_create("my_obj", ipc::shm::InitMode::ForceSmall);
    my_obj->grow_storage(sizeof(MyObj));

    my_obj->age += 1;
    strcpy(my_obj->name, "小明\0");

    cout << "hello " << my_obj->name << ", age " << my_obj->age << endl;
}