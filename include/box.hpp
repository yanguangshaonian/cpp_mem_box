#ifndef IPC_SHM_BOX_HPP
#define IPC_SHM_BOX_HPP
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <map>
#include <string>
#include <cstring>
#include <cstdint>
#include <cassert>
#include <thread>
#include <type_traits>
#include <system_error>
#include <chrono>

// 系统调用头文件
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <pthread.h>
#include <emmintrin.h> // for _mm_pause

namespace ipc {
    // ----------------------------------------------------------------
    // 诊断子系统 (Diagnostics)
    // ----------------------------------------------------------------
    namespace diag {
        inline std::atomic_flag log_spinlock = ATOMIC_FLAG_INIT;

        // 获取线程身份标识 [PID:TID]
        __attribute__((always_inline)) inline void format_thread_id(char* buf, size_t size) {
            static thread_local auto cached_pid = 0;
            static thread_local auto cached_tid = 0;

            // 分支预测优化: 只会进入一次系统调用
            if (__builtin_expect(static_cast<int64_t>(cached_pid == 0), 0) != 0) {
                cached_pid = getpid();
                cached_tid = static_cast<int>(gettid()); // 依赖 _GNU_SOURCE
            }
            snprintf(buf, size, "[%d:%d]", cached_pid, cached_tid);
        }

        // 获取高精度时间戳
        __attribute__((always_inline)) inline void format_timestamp(char* buf, size_t size) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            struct tm tm_info;
            localtime_r(&ts.tv_sec, &tm_info);

            auto len = strftime(buf, size, "%T", &tm_info);
            if (len > 0 && (size_t) len < size) {
                snprintf(buf + len, size - len, ".%03ld", ts.tv_nsec / 1000000);
            }
        }

        // 高性能日志写入 (Direct Syscall Write)
        __attribute__((always_inline)) inline void write(const char* level, const std::string& module,
                                                         const std::string& msg) {
            char buffer[1024];
            char time_buf[32];
            char tid_buf[32];

            format_timestamp(time_buf, sizeof(time_buf));
            format_thread_id(tid_buf, sizeof(tid_buf));

            // 格式化: [Time] [PID:TID] [LEVEL] [Module] Message
            auto len = snprintf(buffer, sizeof(buffer), "[%s] %s [%s] [%s] %s\n", time_buf, tid_buf, level,
                                module.c_str(), msg.c_str());

            // 截断保护
            if (len >= (int) sizeof(buffer)) {
                buffer[sizeof(buffer) - 2] = '\n';
                buffer[sizeof(buffer) - 1] = '\0';
                len = sizeof(buffer) - 1;
            }

            // 自旋锁临界区: 极短, 仅包含 write 系统调用
            while (log_spinlock.test_and_set(std::memory_order_acquire)) {
                _mm_pause();
            }
            ::write(STDOUT_FILENO, buffer, len);
            log_spinlock.clear(std::memory_order_release);
        }
    } // namespace diag

    // ----------------------------------------------------------------
    // 内部实现细节 (Implementation Details)
    // ----------------------------------------------------------------
    namespace detail {
        static __attribute__((always_inline)) inline uint64_t align_up(uint64_t size, uint64_t alignment) {
            return (size + alignment - 1) & ~(alignment - 1);
        }
    } // namespace detail

    // ----------------------------------------------------------------
    // 共享内存核心 (Shared Memory Core)
    // ----------------------------------------------------------------
    namespace shm {
        // 初始化模式配置
        enum class InitMode {
            ForceSmall, // 强制 4KB 页 (Standard Pages)
            ForceLarge  // 强制 2MB 页 (Transparent Huge Pages)
        };

        // ------------------------------------------------------------
        // 健壮锁 (Robust Mutex) - 处理进程崩溃后的死锁恢复
        // ------------------------------------------------------------
        class RobustLock {
            public:
                pthread_mutex_t handle;

                void init() {
                    diag::write("INFO", "RobustLock", "正在初始化健壮互斥锁...");

                    pthread_mutexattr_t attr;
                    pthread_mutexattr_init(&attr);

                    pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
                    pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST);
                    pthread_mutex_init(&this->handle, &attr);

                    auto ret = pthread_mutexattr_destroy(&attr);
                    if (ret != 0) {
                        diag::write("ERROR", "RobustLock", "属性销毁失败: " + std::to_string(ret));
                        throw std::system_error(ret, std::generic_category());
                    }
                    diag::write("INFO", "RobustLock", "初始化完毕 (PTHREAD_MUTEX_ROBUST)");
                }

                // RAII 保护器
                class ScopedGuard {
                    private:
                        RobustLock* lock_;
                        bool acquired_ = false;

                    public:
                        explicit ScopedGuard(RobustLock* l)
                            : lock_(l) {
                            auto ret = pthread_mutex_lock(&this->lock_->handle);
                            if (ret == 0) {
                                this->acquired_ = true;
                            } else if (ret == EOWNERDEAD) {
                                diag::write("WARN", "LockGuard", ">>> 检测到持有者死亡 (EOWNERDEAD) <<<");
                                diag::write("WARN", "LockGuard", "正在恢复锁的一致性 (consistent)...");

                                if (pthread_mutex_consistent(&this->lock_->handle) != 0) {
                                    diag::write("ERROR", "LockGuard", "锁恢复失败!");
                                    throw std::runtime_error("RobustLock recovery failed");
                                }
                                this->acquired_ = true;
                                diag::write("INFO", "LockGuard", "锁已恢复, 继续执行");
                            } else if (ret == ENOTRECOVERABLE) {
                                diag::write("ERROR", "LockGuard", "锁状态不可恢复");
                                throw std::runtime_error("RobustLock ENOTRECOVERABLE");
                            } else {
                                diag::write("ERROR", "LockGuard", "加锁失败: " + std::to_string(ret));
                                throw std::system_error(ret, std::generic_category());
                            }
                        }

                        ~ScopedGuard() {
                            if (this->acquired_) {
                                pthread_mutex_unlock(&this->lock_->handle);
                            }
                        }
                };
        };

        // ------------------------------------------------------------
        // 区域描述符 (Region Descriptor)
        // ------------------------------------------------------------
        class RegionDescriptor {
            public:
                int32_t fd;
                uint8_t* base;
                uint64_t v_size;
        };

        // ------------------------------------------------------------
        // 映射注册表 (Mapping Registry)
        // ------------------------------------------------------------
        class MappingRegistry {
            private:
                // Key: Region End Address
                std::map<uintptr_t, RegionDescriptor> mappings_;
                std::shared_mutex rw_lock_;

                MappingRegistry() = default;

            public:
                MappingRegistry(const MappingRegistry&) = delete;
                MappingRegistry& operator=(const MappingRegistry&) = delete;

                static MappingRegistry& instance() {
                    static MappingRegistry inst;
                    return inst;
                }

                void register_mapping(uint8_t* base, uint64_t size, int32_t fd) {
                    std::unique_lock lock(this->rw_lock_);
                    uintptr_t end = reinterpret_cast<uintptr_t>(base) + size;
                    this->mappings_[end] = {fd, base, size};
                }

                void unregister_mapping(uint8_t* base, uint64_t size) {
                    std::unique_lock lock(this->rw_lock_);
                    uintptr_t end = reinterpret_cast<uintptr_t>(base) + size;
                    this->mappings_.erase(end);
                }

                __attribute__((always_inline)) inline bool find_mapping_fast(void* ptr, RegionDescriptor* out_desc,
                                                                             RegionDescriptor* cache_hint) {
                    uintptr_t p = reinterpret_cast<uintptr_t>(ptr);

                    // Fast Path: TLS Cache
                    if (cache_hint != nullptr && cache_hint->base != nullptr) {
                        uintptr_t start = reinterpret_cast<uintptr_t>(cache_hint->base);
                        uintptr_t end = start + cache_hint->v_size;
                        if (p >= start && p < end) {
                            *out_desc = *cache_hint;
                            return true;
                        }
                    }

                    // Slow Path: Map Lookup
                    std::shared_lock lock(this->rw_lock_);
                    auto it = this->mappings_.upper_bound(p);

                    if (it != this->mappings_.end()) {
                        uintptr_t base = reinterpret_cast<uintptr_t>(it->second.base);
                        if (p >= base) {
                            *out_desc = it->second;
                            if (cache_hint != nullptr) {
                                *cache_hint = it->second;
                            }
                            return true;
                        }
                    }
                    return false;
                }
        };

        // ------------------------------------------------------------
        // 内存控制块 (Control Block)
        // ------------------------------------------------------------
        static constexpr auto IPC_SHM_SIGNATURE = 0xABCDEFABCDEF1996;
        static constexpr auto VIRTUAL_RESERVATION_SIZE = 128ULL * 1024 * 1024 * 1024; // 128GB

        class alignas(256) ControlBlock {
            public:
                volatile std::atomic<uint64_t> signature; // 文件签名 (Magic)
                uint64_t page_size;                       // 页对齐大小
                std::atomic<uint64_t> capacity_bytes;     // 当前提交的物理容量
                uint64_t payload_offset;                  // 有效载荷偏移量
                RobustLock extend_lock;                   // 扩容专用锁
        };

        // ------------------------------------------------------------
        // 物理页提交器 (Page Committer)
        // ------------------------------------------------------------
        class PageCommitter {
            public:
                static __attribute__((always_inline)) inline bool commit(void* payload_ptr, uint64_t required_bytes) {
                    static thread_local RegionDescriptor tls_desc = {0, nullptr, 0};
                    RegionDescriptor desc;

                    // 查找指针所属区域
                    if (!MappingRegistry::instance().find_mapping_fast(payload_ptr, &desc, &tls_desc)) {
                        diag::write("ERROR", "Committer", "Commit失败: 指针未受管辖");
                        return false;
                    }

                    auto* cb = reinterpret_cast<ControlBlock*>(desc.base);
                    auto total_needed = cb->payload_offset + required_bytes;

                    // 乐观无锁检查 (Acquire)
                    if (cb->capacity_bytes.load(std::memory_order_acquire) >= total_needed) {
                        return true;
                    }

                    // 加锁扩容 (悲观路径)
                    RobustLock::ScopedGuard guard(&cb->extend_lock);

                    // Double Check
                    auto current_cap = cb->capacity_bytes.load(std::memory_order_relaxed);
                    if (current_cap >= total_needed) {
                        return true;
                    }

                    // 计算对齐
                    auto align = cb->page_size != 0 ? cb->page_size : 4096;
                    auto new_cap = detail::align_up(total_needed, align);

                    diag::write("INFO", "Committer",
                                "扩展物理内存: " + std::to_string(current_cap) + " -> " + std::to_string(new_cap));

                    if (ftruncate(desc.fd, static_cast<off_t>(new_cap)) != 0) {
                        diag::write("ERROR", "Committer", "ftruncate 失败: " + std::string(strerror(errno)));
                        return false;
                    }

                    // 提交新容量 (Release)
                    cb->capacity_bytes.store(new_cap, std::memory_order_release);
                    return true;
                }
        };

        // ------------------------------------------------------------
        // 共享对象基类 (CRTP)
        // ------------------------------------------------------------
        template<typename Derived>
        class Boxed {
            protected:
                Boxed() {
#ifndef NDEBUG
                    RegionDescriptor desc;
                    if (!MappingRegistry::instance().find_mapping_fast(this, &desc, nullptr)) {
                        diag::write("ERROR", "ShmBase", "对象未位于托管的共享内存区域内");
                        std::abort();
                    }
#endif
                }

                ~Boxed() = default;

            public:
                // 必须通过 Box 的 Placement New 在共享内存上创建
                void* operator new(size_t) = delete;
                void* operator new[](size_t) = delete;
                void operator delete(void*) = delete;
                void operator delete[](void*) = delete;

                // Placement New (Box 专用通道)
                void* operator new(size_t, void* ptr) {
                    return ptr;
                }

                void operator delete(void*, void*) {}

                // 自我扩容
                __attribute__((always_inline)) inline bool grow_storage(uint64_t bytes_needed) {
                    return PageCommitter::commit(static_cast<Derived*>(this), bytes_needed);
                }
        };

        // ------------------------------------------------------------
        // 共享对象 (Box Object)
        // ------------------------------------------------------------
        template<class T>
        class Box {
                static_assert(std::is_trivially_copyable_v<T>, "T 必须可以安全的复制");
                static_assert(std::is_base_of_v<Boxed<T>, T>, "Box<T>: T 必须继承自 ipc::shm::Boxed<T>");

            private:
                std::string name_;
                int32_t fd_ = -1;
                uint8_t* base_ = nullptr;
                ControlBlock* cb_ = nullptr;
                T* payload_ = nullptr;

                // 内部函数: 挂载逻辑 (事务性修复版)
                bool mount_region(bool is_creator, InitMode mode) {
                    diag::write("INFO", "Mount", "映射虚拟地址空间...");
                    // 使用局部变量持有资源, 绝不污染 this->base_
                    void* temp_ptr =
                        mmap(nullptr, VIRTUAL_RESERVATION_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, this->fd_, 0);

                    if (temp_ptr == MAP_FAILED) {
                        diag::write("ERROR", "Mount", "mmap 失败: " + std::string(strerror(errno)));
                        return false;
                    }

                    uint8_t* temp_base = static_cast<uint8_t*>(temp_ptr);
                    auto* temp_cb = reinterpret_cast<ControlBlock*>(temp_base);
                    bool success = false;

                    // 使用 do-while(0) 结构统一管理错误跳出
                    do {
                        if (is_creator) {
                            diag::write("INFO", "Mount", "[创建者] 初始化控制块...");
                            try {
                                temp_cb->extend_lock.init();
                            } catch (const std::exception& e) {
                                diag::write("ERROR", "Mount", "锁初始化失败... " + std::string(e.what()));
                                break; // 跳出，执行 munmap
                            }

                            auto sys_page = sysconf(_SC_PAGESIZE);
                            constexpr auto huge_page = 1024 * 1024 * 2;
                            bool use_huge = (mode == InitMode::ForceLarge);

                            if (use_huge) {
                                madvise(temp_ptr, VIRTUAL_RESERVATION_SIZE, MADV_HUGEPAGE);
                                temp_cb->page_size = huge_page;
                            } else {
                                madvise(temp_ptr, VIRTUAL_RESERVATION_SIZE, MADV_NOHUGEPAGE);
                                temp_cb->page_size = sys_page;
                            }

                            temp_cb->payload_offset = detail::align_up(sizeof(ControlBlock), alignof(T));
                            auto init_sz = detail::align_up(temp_cb->payload_offset + sizeof(T), temp_cb->page_size);

                            if (ftruncate(this->fd_, static_cast<int64_t>(init_sz)) != 0) {
                                diag::write("ERROR", "Mount", "初始 ftruncate 失败");
                                break;
                            }
                            temp_cb->capacity_bytes.store(init_sz, std::memory_order_relaxed);
                        } else {
                            diag::write("INFO", "Mount", "[附加者] 等待签名...");
                            auto spins = 0;
                            while (temp_cb->signature.load(std::memory_order_acquire) != IPC_SHM_SIGNATURE) {
                                if (spins < 1000)
                                    _mm_pause();
                                else if (spins < 3000)
                                    std::this_thread::yield();
                                else
                                    std::this_thread::sleep_for(std::chrono::milliseconds(1));

                                if (spins > 5000) {
                                    diag::write("ERROR", "Mount", "等待签名超时");
                                    break;
                                }
                                spins += 1;
                            }
                            if (temp_cb->signature.load(std::memory_order_relaxed) != IPC_SHM_SIGNATURE) {
                                break;
                            }
                        }
                        success = true;
                    } while (false);

                    if (!success) {
                        // 事务回滚: 释放刚才申请的 mmap
                        munmap(temp_base, VIRTUAL_RESERVATION_SIZE);
                        return false;
                    }

                    // 事务提交: 一切成功后, 才赋值给成员变量
                    this->base_ = temp_base;
                    this->cb_ = temp_cb;
                    this->payload_ = reinterpret_cast<T*>(this->base_ + this->cb_->payload_offset);

                    MappingRegistry::instance().register_mapping(this->base_, VIRTUAL_RESERVATION_SIZE, this->fd_);
                    return true;
                }

            public:
                // --------------------------------------------------------
                // API: 创建或连接 (Attach or Create)
                // --------------------------------------------------------
                template<typename... Args>
                bool attach_or_create(const std::string& name, InitMode mode = InitMode::ForceLarge, Args&&... args) {
                    if (this->fd_ >= 0 || this->base_ != nullptr) {
                        diag::write("ERROR", "Box", "Box 已经处于 Open 状态，请先析构或重置");
                        return false;
                    }
                    this->name_ = name;
                    bool is_creator = false;

                    diag::write("INFO", "Box", "打开共享对象: " + this->name_);

                    // 尝试原子创建
                    this->fd_ = shm_open(this->name_.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0660);
                    if (this->fd_ >= 0) {
                        is_creator = true;
                        diag::write("INFO", "Box", "原子创建成功 (Owner)");
                    } else if (errno == EEXIST) {
                        diag::write("WARN", "Box", "对象已存在, 尝试 Attach...");
                        this->fd_ = shm_open(this->name_.c_str(), O_RDWR | O_CLOEXEC, 0660);
                        if (this->fd_ < 0) {
                            diag::write("ERROR", "Box", "Attach 失败: " + std::string(strerror(errno)));
                            return false;
                        }
                        diag::write("INFO", "Box", "Attach 句柄打开成功");
                    } else {
                        diag::write("ERROR", "Box", "shm_open 致命错误: " + std::string(strerror(errno)));
                        return false;
                    }

                    if (!mount_region(is_creator, mode)) {
                        close(this->fd_);
                        this->fd_ = -1;
                        if (is_creator) {
                            shm_unlink(this->name_.c_str());
                        }
                        return false;
                    }

                    if (is_creator) {
                        try {
                            new (this->payload_) T(std::forward<Args>(args)...);
                        } catch (const std::exception& e) {
                            diag::write("ERROR", "Box", "Payload 构造异常(std): " + std::string(e.what()));
                            shm_unlink(this->name_.c_str());
                            throw;
                        } catch (...) {
                            diag::write("ERROR", "Box", "Payload 构造异常(unknown)!");
                            shm_unlink(this->name_.c_str());
                            throw;
                        }

                        std::atomic_thread_fence(std::memory_order_release);
                        this->cb_->signature.store(IPC_SHM_SIGNATURE, std::memory_order_release);
                        diag::write("INFO", "Box", "Payload 构造完成, 服务已发布 (Signature Written)");
                    }
                    return true;
                }

                // --------------------------------------------------------
                // Accessors
                // --------------------------------------------------------
                __attribute__((always_inline, pure)) inline T* __restrict__ ptr() {
                    return this->payload_;
                }

                __attribute__((always_inline, pure)) inline T* __restrict__ operator->() {
                    return this->payload_;
                }

                __attribute__((always_inline, pure)) inline const T* __restrict__ operator->() const {
                    return this->payload_;
                }

                ~Box() {
                    if (this->base_ != nullptr) {
                        MappingRegistry::instance().unregister_mapping(this->base_, VIRTUAL_RESERVATION_SIZE);
                        munmap(this->base_, VIRTUAL_RESERVATION_SIZE);
                    }
                    if (this->fd_ >= 0) {
                        close(this->fd_);
                    }
                }
        };
    } // namespace shm
} // namespace ipc
#endif // IPC_SHM_BOX_HPP


/*
透明的扩容, 智能句柄

namespace ipc {
namespace shm {

    template <typename T>
    class ShmHandle {
    private:
        T* ptr_; // 仅持有一个指针, sizeof(ShmHandle) == 8 bytes

    public:
        // 构造函数
        explicit ShmHandle(T* p) : ptr_(p) {}

        // 1. 像指针一样访问成员 (零开销)
        __attribute__((always_inline)) T* operator->() { return ptr_; }
        __attribute__((always_inline)) const T* operator->() const { return ptr_; }
        __attribute__((always_inline)) T& operator*() { return *ptr_; }

        // 2. 核心功能: 扩容
        // 返回 true 表示成功, false 表示失败
        bool resize(uint64_t required_bytes) {
            // 直接调用底层 PageCommitter
            return PageCommitter::commit(ptr_, required_bytes);
        }

        // 3. 允许隐式转换为裸指针 T*
        // 这样你可以把 handle 直接传给接受 T* 的老函数
        operator T*() { return ptr_; }
        operator const T*() const { return ptr_; }

        // 获取裸指针（显式）
        T* get() { return ptr_; }
    };

} // namespace shm
} // namespace ipc



template<class T>
class Box {
    // ... 原有代码 ...

public:
    // ... 原有代码 ...

    // 新增: 获取智能句柄
    ShmHandle<T> handle() {
        return ShmHandle<T>(this->payload_);
    }
};
*/