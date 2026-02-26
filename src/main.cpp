// #include <cstdint>
// #include <cstdlib>
// #include <iostream>
// #include <string>
// #include <vector>
// #include <random>
// #include <cstring>
// #include <chrono>
// #include <algorithm>
// #include <iomanip>
// #include "box.hpp"

// using namespace std;

// class MyObj : public ipc::shm::Boxed<MyObj> {
//     public:
//         uint64_t age1;
//         uint64_t age;
//         char name[40];
// };

// int main(const int argc, char* argv[]) {

//     auto my_obj = ipc::shm::Box<MyObj>{};
//     my_obj.attach_or_create("my_obj");
//     my_obj->grow_storage(sizeof(MyObj));

//     my_obj->age += 1;
//     strcpy(my_obj->name, "小明\0");

//     cout << "hello " << my_obj->name << ", age " << my_obj->age << endl;
// }


#include "box.hpp"
#include <cstddef>
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>

// 柔性数组业务结构定义
// Flexible array business structure definition
struct FamDemo : public ipc::shm::Boxed<FamDemo> {
        uint64_t current_items;

        // 柔性数组: 必须是标准布局结构体的最后一个成员
        // Flexible array: Must be the last member of a Standard Layout struct
        uint64_t data[0];
};

int main() {
    ipc::shm::Box<FamDemo> box;

    // 为了让 RSS 变化极其明显，这里使用 ForceSmall (4KB页)
    // To make RSS changes extremely obvious, ForceSmall (4KB pages) is used here
    if (!box.attach_or_create("/shm_fam_demo_v1", ipc::shm::InitMode::ForceSmall)) {
        std::cerr << "Failed to initialize SHM Box." << std::endl;
        return 1;
    }

    auto pid = getpid();
    std::cout << "==================================================\n";
    std::cout << "[INFO] 进程启动成功 / Process started successfully\n";
    std::cout << "[INFO] PID: " << pid << "\n";
    std::cout << "请在另一个终端执行以下命令进行观测:\n";
    std::cout << "watch -n 1 \"ps -o pid,vsz,rss,comm -p " << pid << "\"\n";
    std::cout << "==================================================\n";

    // 设置扩容目标为 20 * 1024 * 1024 个元素，约 160 MB 物理内存
    // Set expansion target to 20M elements, approx 160 MB physical memory
    const uint64_t target_items = static_cast<const uint64_t>(20 * 1024 * 1024);
    const uint64_t large_size = sizeof(FamDemo) + target_items * sizeof(uint64_t);
    const uint64_t small_size = sizeof(FamDemo);

    auto cycle_count = 0;

    while (true) {
        cycle_count += 1;
        std::cout << "\n--- Cycle " << cycle_count << " ---\n";

        // ----------------------------------------------------------------
        // 阶段 1: 扩容并强制分配物理页 (Phase 1: Grow & Force Page Allocation)
        // ----------------------------------------------------------------
        std::cout << "[GROW] 正在扩容到底层文件大小: 160MB...\n";
        box->grow_storage(large_size);
        box->current_items = target_items;

        std::cout << "[GROW] 正在写入内存以触发缺页中断 (按需分页/Demand Paging)...\n";
        // 核心观测点：必须写入才能导致 RSS 增长
        // Core observation point: Writing is mandatory to cause RSS growth
        for (uint64_t i = 0; i < target_items; i += 1) {
            box->data[i] = i;
        }
        std::cout << "[GROW] 写入完成，此时 RSS 应处于高位 (~160000 KB)。休眠 5 秒...\n";
        std::this_thread::sleep_for(std::chrono::seconds(5));

        // ----------------------------------------------------------------
        // 阶段 2: 缩容释放物理页 (Phase 2: Shrink & Release Physical Pages)
        // ----------------------------------------------------------------
        std::cout << "[SHRINK] 正在截断文件，将物理页归还操作系统...\n";
        box->shrink_storage(small_size);
        box->current_items = 0;

        std::cout << "[SHRINK] 截断完成，此时 RSS 应处于低位 (接近 0 KB)。休眠 5 秒...\n";
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }

    return 0;
}