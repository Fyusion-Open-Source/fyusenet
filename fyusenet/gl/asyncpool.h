//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Thread-Pool for Asynchronous (GL) Operations (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <thread>
#include <list>
#include <mutex>
#include <atomic>
#include <chrono>
#include <functional>
#include <condition_variable>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../gpu/gfxcontextlink.h"

//------------------------------------------ Constants ---------------------------------------------


namespace fyusion {
namespace opengl {
//------------------------------------- Public Declarations ----------------------------------------

/**
 * @brief Thread-pool class for asynchronous GL (and non-GL) processing
 *
 * This class represents a simple thread-pool that supports multiple (shared) GL contexts for
 * multi-threaded issue of OpenGL commands to the GPU. Each GL thread in the thread-pool is
 * associated with a GL context, which is usually a context that is derived from a main context,
 * such that all the contexts have shared resources.
 *
 * In addition to GL based contexts, this pool also supports non-GL threads which are basically
 * the same as the GL threads, just without any associated context.
 *
 * The thread objects that are returned by this pool are to be used as simple task execution
 * units using C++ function wrappers. For example:
 *
 * @code
 *   AsyncPool::GLThread thread = AsyncPool::getContextThread(someContext);
 *   assert(thread.isValid());
 *   auto fancycode = [&]() { int a = 1+1; };
 *   thread->waitTask(fancycode);
 * @endcode
 *
 * This will execute the supplied \c fancycode function and wait for the execution to finalize.
 * In order to execute a task without blocking, use the setTask() method instead.
 */
class AsyncPool {

    /**
     * @brief Base class for pool thread
     *
     * This class consists of a basic task-based thread implementation. Once created, the thread
     * will wait for tasks to be assigned to it and execute them. It contains blocking and non-
     * blocking modes of execution. This particular base class implementation has no OpenGL
     * specifics, check GLThreadImpl for GL specifics.
     *
     * @see GLThreadImpl
     */
    class ThreadImpl {
        friend class AsyncPool;
    public:
       bool setTask(std::function<void()> func);
       bool waitTask(std::function<void()> func);
       void wait();
       bool isBusy() const;
    protected:
       ThreadImpl();
       virtual ~ThreadImpl();
       std::thread * thread_ = nullptr;                 //!< Pointer to actual thread
       mutable std::mutex issueLock_;                   //!< Lock that prevents issueing tasks in parallel
       std::mutex lock_;                                //!< Lock that is used for #waiting_
       std::mutex taskDoneMutex_;                       //!< Lock that is used for #taskDone_
       std::condition_variable taskDone_;               //!< Wait condition that will wake up when a task is done
       std::condition_variable waiting_;                //!< Wait condition that will wake up the pool-thread when a task is available
       std::function<void()> task_;                     //!< Task to execute in the thread
       static std::function<void()> EMPTY;              //!< Placeholder for an empty task
       bool quit_ = true;                               //!< Loop condition, when set to \c true, thread-loop will exit#
       /**
        * @brief Timestamp when the thread was last used to execute a task
        */
       std::chrono::time_point<std::chrono::steady_clock> lastUsed_;
    };


    /**
     * @brief GL-specific pool thread
     *
     * This class expands the ThreadImpl class by a GL context that is assigned to it and which
     * is the active context in that thread.
     *
     * @see GfxContextLink, ThreadImpl
     */
    class GLThreadImpl : public ThreadImpl {
        friend class AsyncPool;
        friend class GLThread;
     protected:
        GLThreadImpl(const fyusenet::GfxContextLink& ctx);
        fyusenet::GfxContextLink context_;                     //!< Link to GL context that is current to this thread
    };


    /**
     * @brief Data structure that represents a single entry in the thread pool
     *
     * Stores a pointer to the actual thread as well as a reference count.
     *
     * @todo Use shared_ptr instead
     */
    struct entry {
        entry(ThreadImpl *t, int ref) : thread(t) {
            refcount.store(ref);
        }
        entry(entry && src) {
            thread = src.thread;
            // NOTE (mw) not atomic
            refcount.store(src.refcount.load());
        }
        ThreadImpl * thread = nullptr;
        std::atomic<uint32_t> refcount{0};
    };

 public:

    /**
     * @brief Thread handle (w/o associated GL context)
     *
     * This class provides a handle to a ThreadImpl instance and augments it with reference
     * counting. Dereference objects of this class to get access to the ThreadImpl interface
     * in order to run tasks on a thread.
     *
     * @note Thread objects should only be held for as long as necessary, as pending objects
     *       may deplete the pool. A thread will not be re-used until the task has finished.
     */
    class Thread {
        friend class AsyncPool;

        /**
         * @brief Constructor
         *
         * @param thread Actual thread that is wrapped by this instance
         * @param ref Pointer to reference counter to the actual thread
         */
        Thread(ThreadImpl *thread, std::atomic<uint32_t> * ref) : thread_(thread), refcount_(ref) {
            assert(thread_);
            assert(refcount_);
            assert(refcount_->load() == 1);
        }

     public:

        /**
         * @brief Empty constructor
         */
        Thread() {
        }

        /**
         * @brief Copy constructor
         *
         * @param src Object to copy contents from
         */
        Thread(const Thread& src) {
            thread_ = src.thread_;
            refcount_= src.refcount_;
            if (refcount_) refcount_->fetch_add(1);
        }

        /**
         * @brief Destructor
         */
        ~Thread() {
            if (thread_) {
                int cnt = (int)refcount_->fetch_sub(1);
                assert(cnt > 0);
                thread_ = nullptr;
            }
        }

        /**
         * @brief Reset thread handle to invalid state
         */
        void reset() {
            if (thread_) {
                int cnt = (int)refcount_->fetch_sub(1);
                assert(cnt > 0);
                thread_ = nullptr;
                refcount_ = nullptr;
            }
         }

        /**
         * @brief Assignment operator
         *
         * @param src Object to copy contents from
         *
         * @return Reference to current object
         */
        Thread & operator=(const Thread& src) {
            if (this == &src) return *this;
            auto * oldref = refcount_;
            thread_ = src.thread_;
            refcount_= src.refcount_;
            if (refcount_) refcount_->fetch_add(1);
            if (oldref) oldref->fetch_sub(1);
            return *this;
        }


        /**
         * @brief Dereference
         *
         * @return Pointer to underlying ThreadImpl
         */
        ThreadImpl * operator->() {
            return thread_;
        }


        /**
         * @brief Dereference
         *
         * @return Pointer to underlying ThreadImpl
         */
        ThreadImpl * operator*() {
            return thread_;
        }


        /**
         * @brief Check if thread is valid
         *
         * @retval true if thread is valid
         * @retval false otherwise
         */
        bool isValid() const {
            return (thread_) && (refcount_);
        }

     protected:
        ThreadImpl * thread_ = nullptr;                 //!< Actual thread to use
        std::atomic<uint32_t> * refcount_ = nullptr;    //!< Reference count to the underlying thread
    };


    /**
     * @brief Thread handle (w/o associated GL context)
     *
     * This class provides a handle to a ThreadImpl instance and augments it with reference
     * counting. Dereference objects of this class to get access to the ThreadImpl interface
     * in order to run tasks on a thread.
     *
     * @note Thread objects should only be held for as long as necessary, as pending objects
     *       may deplete the pool. A thread will not be re-used until the task has finished.
     */
    class GLThread : public Thread {
        friend class AsyncPool;

        /**
         * @brief Constructor
         *
         * @param thread Actual thread that is wrapped by this instance
         * @param ref Pointer to reference counter to the actual thread
         */
        GLThread(GLThreadImpl *thread, std::atomic<uint32_t> * ref) : Thread(thread, ref) {
        }
     public:
        GLThread() {
        }

        /**
         * @brief Retrieve link to GL context current to the thread object
         *
         * @return Link to GL context that is bound to the thread
         */
        const fyusenet::GfxContextLink & context() const {
            if (!thread_) return fyusenet::GfxContextLink::EMPTY;
            else return ((GLThreadImpl *)thread_)->context_;
        }
    };

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    static Thread getThread();
    static GLThread getDerivedContextThread(const fyusenet::GfxContextLink& ctx, int timeout=-1);
    static GLThread getContextThread(const fyusenet::GfxContextLink& ctx, int timeout=-1);
    static void createDerivedBatch(const fyusenet::GfxContextLink& ctx, int numThreads);
    static void tearDown();
    static void setMaxGLThreads(int mt);
    static void logStatistics();
    static bool isEmpty();

 private:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    static void sniff();
    static bool makeCurrent(const fyusenet::GfxContextLink& ctx);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    static bool goingDown_;                             //!< Indicator that pool is being torn down
    static std::mutex lock_;                            //!< Misc serialization lock
    static std::atomic<int> teardownProtection_;        //!< Atomic counter to protect against teardown while a thread is being obtained
    static std::list<entry> glThreads_;                 //!< List of threads with associated GL contexts
    static std::list<entry> threads_;                   //!< List of threads without associated GL contexts
    static std::thread * watchDog_;                     //!< Watchdog thread that reaps inactive threads
    static int maxGLThreads_;                           //!< Limit on number of GL threads
    static uint64_t waitCycles_;                        //!< For debug & statistics, the number of busy-waiting rounds until a thread was available
    static uint64_t immediateHit_;                      //!< For debug & statistics, number of thread requests that were successful without waiting
    static uint64_t requests_;                          //!< For debug & statistics, total number of thread requests
    static uint32_t reapCount_;                         //!< For debug & statistics, number of thread reap operations
#ifdef ANDROID
    constexpr static int HARD_MAX_THREADS = 32;         //!< Hard maximum in the number of threads for #threads_ and #glThreads_ (in combination)
#else
    constexpr static int HARD_MAX_THREADS = 128;        //!< Hard maximum in the number of threads for #threads_ and #glThreads_ (in combination)
#endif
    constexpr static int INACTIVITY_TIMEOUT = 15000;    //!< Milliseconds of idle-time for a (non-context) thread to be reaped
};

} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
