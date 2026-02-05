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
            static thread_local int cached_pid = 0;
            static thread_local int cached_tid = 0;

            // 分支预测优化：只会进入一次系统调用
            if (__builtin_expect(cached_pid == 0, 0)) {
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

            int len = strftime(buf, size, "%T", &tm_info);
            if (len > 0 && (size_t) len < size) {
                snprintf(buf + len, size - len, ".%03ld", ts.tv_nsec / 1000000);
            }
        }

        // 高性能日志写入 (Direct Syscall Write)
        __attribute__((always_inline)) inline void write(const char* level, const std::string& module, const std::string& msg) {
            char buffer[1024]; // 栈上分配，避免 malloc
            char time_buf[32];
            char tid_buf[32];

            format_timestamp(time_buf, sizeof(time_buf));
            format_thread_id(tid_buf, sizeof(tid_buf));

            // 格式化: [Time] [PID:TID] [LEVEL] [Module] Message
            int len = snprintf(buffer, sizeof(buffer), "[%s] %s [%s] [%s] %s\n", time_buf, tid_buf, level,
                               module.c_str(), msg.c_str());

            // 截断保护
            if (len >= (int) sizeof(buffer)) {
                buffer[sizeof(buffer) - 2] = '\n';
                buffer[sizeof(buffer) - 1] = '\0';
                len = sizeof(buffer) - 1;
            }

            // 自旋锁临界区：极短，仅包含 write 系统调用
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

                    int ret = pthread_mutexattr_destroy(&attr);
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
                            int ret = pthread_mutex_lock(&this->lock_->handle);
                            if (ret == 0) {
                                this->acquired_ = true;
                            } else if (ret == EOWNERDEAD) {
                                diag::write("WARN", "LockGuard", ">>> 检测到持有者死亡 (EOWNERDEAD) <<<");
                                diag::write("WARN", "LockGuard", "正在恢复锁的一致性 (consistent)...");

                                if (pthread_mutex_consistent(&this->lock_->handle) != 0) {
                                    diag::write("FATAL", "LockGuard", "锁恢复失败!");
                                    throw std::runtime_error("RobustLock recovery failed");
                                }
                                this->acquired_ = true;
                                diag::write("INFO", "LockGuard", "锁已恢复，继续执行");
                            } else if (ret == ENOTRECOVERABLE) {
                                diag::write("FATAL", "LockGuard", "锁状态不可恢复");
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
                int fd;
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

                void register_mapping(uint8_t* base, uint64_t size, int fd) {
                    std::unique_lock lock(this->rw_lock_);
                    uintptr_t end = reinterpret_cast<uintptr_t>(base) + size;
                    this->mappings_[end] = {fd, base, size};
                }

                void unregister_mapping(uint8_t* base, uint64_t size) {
                    std::unique_lock lock(this->rw_lock_);
                    uintptr_t end = reinterpret_cast<uintptr_t>(base) + size;
                    this->mappings_.erase(end);
                }

                 __attribute__((always_inline)) inline bool find_mapping_fast(void* ptr, RegionDescriptor* out_desc, RegionDescriptor* cache_hint) {
                    uintptr_t p = reinterpret_cast<uintptr_t>(ptr);

                    // 1. Fast Path: TLS Cache
                    if (cache_hint && cache_hint->base) {
                        uintptr_t start = reinterpret_cast<uintptr_t>(cache_hint->base);
                        uintptr_t end = start + cache_hint->v_size;
                        if (p >= start && p < end) {
                            *out_desc = *cache_hint;
                            return true;
                        }
                    }

                    // 2. Slow Path: Map Lookup
                    std::shared_lock lock(this->rw_lock_);
                    auto it = this->mappings_.upper_bound(p);

                    if (it != this->mappings_.end()) {
                        uintptr_t base = reinterpret_cast<uintptr_t>(it->second.base);
                        if (p >= base) {
                            *out_desc = it->second;
                            if (cache_hint) {
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
        static constexpr uint64_t IPC_SHM_SIGNATURE = 0xABCDEFABCDEF1996;
        static constexpr uint64_t VIRTUAL_RESERVATION_SIZE = 128ULL * 1024 * 1024 * 1024; // 128GB

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
                static  __attribute__((always_inline)) inline bool commit(void* payload_ptr, uint64_t required_bytes) {
                    static thread_local RegionDescriptor tls_desc = {0, nullptr, 0};
                    RegionDescriptor desc;

                    // 查找指针所属区域
                    if (!MappingRegistry::instance().find_mapping_fast(payload_ptr, &desc, &tls_desc)) {
                        diag::write("ERROR", "Committer", "Commit失败: 指针未受管辖");
                        return false;
                    }

                    auto* cb = reinterpret_cast<ControlBlock*>(desc.base);
                    uint64_t total_needed = cb->payload_offset + required_bytes;

                    // 1. 乐观无锁检查 (Acquire)
                    if (cb->capacity_bytes.load(std::memory_order_acquire) >= total_needed) {
                        return true;
                    }

                    // 2. 加锁扩容 (悲观路径)
                    RobustLock::ScopedGuard guard(&cb->extend_lock);

                    // Double Check
                    uint64_t current_cap = cb->capacity_bytes.load(std::memory_order_relaxed);
                    if (current_cap >= total_needed) {
                        return true;
                    }

                    // 计算对齐
                    uint64_t align = cb->page_size ? cb->page_size : 4096;
                    uint64_t new_cap = detail::align_up(total_needed, align);

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
        // 共享对象 (Box Object)
        // ------------------------------------------------------------
        template<class T>
        class Box {
                static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

            private:
                std::string name_;
                int fd_ = -1;
                uint8_t* base_ = nullptr;
                ControlBlock* cb_ = nullptr;
                T* payload_ = nullptr;

                bool mount_region(bool is_creator, InitMode mode) {
                    diag::write("INFO", "Mount",
                                "映射虚拟地址空间 (" + std::to_string(VIRTUAL_RESERVATION_SIZE / 1024 / 1024 / 1024) +
                                    "GB)...");

                    int prot = PROT_READ | PROT_WRITE; // 总是读写
                    void* ptr = mmap(nullptr, VIRTUAL_RESERVATION_SIZE, prot, MAP_SHARED, this->fd_, 0);

                    if (ptr == MAP_FAILED) {
                        diag::write("ERROR", "Mount", "mmap 失败: " + std::string(strerror(errno)));
                        return false;
                    }

                    this->base_ = static_cast<uint8_t*>(ptr);
                    this->cb_ = reinterpret_cast<ControlBlock*>(base_);

                    if (is_creator) {
                        diag::write("INFO", "Mount", "[创建者] 初始化控制块 (ControlBlock)...");

                        this->cb_->extend_lock.init();

                        uint64_t sys_page = sysconf(_SC_PAGESIZE);
                        uint64_t huge_page = 2 * 1024 * 1024;
                        bool use_huge = (mode == InitMode::ForceLarge);

                        // 建议内核使用透明大页
                        if (use_huge) {
                            madvise(ptr, VIRTUAL_RESERVATION_SIZE, MADV_HUGEPAGE);
                            this->cb_->page_size = huge_page;
                            diag::write("INFO", "Mount", "启用大页模式 (HugePage 2MB)");
                        } else {
                            madvise(ptr, VIRTUAL_RESERVATION_SIZE, MADV_NOHUGEPAGE);
                            this->cb_->page_size = sys_page;
                            diag::write("INFO", "Mount", "使用标准页模式 (4KB)");
                        }

                        // 计算 Payload 偏移
                        this->cb_->payload_offset = detail::align_up(sizeof(ControlBlock), alignof(T));

                        // 初始提交
                        uint64_t init_sz = detail::align_up(this->cb_->payload_offset + sizeof(T), this->cb_->page_size);

                        if (ftruncate(this->fd_, init_sz) != 0) {
                            diag::write("ERROR", "Mount", "初始 ftruncate 失败");
                            return false;
                        }

                        this->cb_->capacity_bytes.store(init_sz, std::memory_order_relaxed);
                        diag::write("INFO", "Mount", "初始物理内存已提交: " + std::to_string(init_sz) + " bytes");

                        // 内存屏障 & 签名写入
                        std::atomic_thread_fence(std::memory_order_release);
                        this->cb_->signature.store(IPC_SHM_SIGNATURE, std::memory_order_release);
                        diag::write("INFO", "Mount", "签名已写入，服务就绪");

                    } else {
                        diag::write("INFO", "Mount", "[附加者] 等待控制块签名...");

                        int spins = 0;
                        while (this->cb_->signature.load(std::memory_order_acquire) != IPC_SHM_SIGNATURE) {
                            if (spins < 1000) {
                                _mm_pause();
                            } else if (spins < 3000) {
                                std::this_thread::yield();
                            } else {
                                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                            }

                            if (spins > 5000) {
                                diag::write("ERROR", "Mount", "等待签名超时 (>5s)");
                                return false;
                            }
                            spins += 1;
                        }
                        diag::write("INFO", "Mount", "签名校验通过，连接成功");
                    }

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
                    this->name_ = name;
                    bool is_creator = false;

                    diag::write("INFO", "Box", "打开共享对象: " + this->name_);

                    // 尝试原子创建
                    this->fd_ = shm_open(this->name_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0660);
                    if (this->fd_ >= 0) {
                        is_creator = true;
                        diag::write("INFO", "Box", "原子创建成功 (Owner)");
                    } else if (errno == EEXIST) {
                        diag::write("WARN", "Box", "对象已存在，尝试 Attach...");
                        this->fd_ = shm_open(this->name_.c_str(), O_RDWR, 0660);
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
                        if (is_creator) {
                            {
                                shm_unlink(this->name_.c_str());
                            }
                        }
                        return false;
                    }

                    if (is_creator) {
                        diag::write("INFO", "Box", "构造 Payload 对象...");
                        new (this->payload_) T(std::forward<Args>(args)...);
                    }
                    return true;
                }

                // --------------------------------------------------------
                // API: 提交物理内存 (Commit)
                // --------------------------------------------------------
                 __attribute__((always_inline)) inline bool commit(uint64_t used_bytes) {
                    return PageCommitter::commit(this->payload_, used_bytes);
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
                    if (this->base_) {
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