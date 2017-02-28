
#ifndef __GPUDMAIOTCL_H__
#define __GPUDMAIOTCL_H__

//-----------------------------------------------------------------------------

#define GPUMEM_DRIVER_NAME             "gpumem"

//-----------------------------------------------------------------------------

#ifdef __linux__
#include <linux/types.h>
#ifndef __KERNEL__
#include <sys/ioctl.h>
#endif
#define GPUMEM_DEVICE_TYPE             'g'
#define GPUMEM_MAKE_IOCTL(c) _IO(GPUMEM_DEVICE_TYPE, (c))
#endif

#define IOCTL_GPUMEM_LOCK		GPUMEM_MAKE_IOCTL(10)
#define IOCTL_GPUMEM_UNLOCK		GPUMEM_MAKE_IOCTL(11)
#define IOCTL_GPUMEM_STATE		GPUMEM_MAKE_IOCTL(12)

//-----------------------------------------------------------------------------
// for boundary alignment requirement
#define GPU_BOUND_SHIFT 16
#define GPU_BOUND_SIZE ((u64)1 << GPU_BOUND_SHIFT)
#define GPU_BOUND_OFFSET (GPU_BOUND_SIZE-1)
#define GPU_BOUND_MASK (~GPU_BOUND_OFFSET)

//-----------------------------------------------------------------------------

struct gpudma_lock_t {
    void*    handle;
    uint64_t addr;
    uint64_t size;
    size_t   page_count;
};

//-----------------------------------------------------------------------------

struct gpudma_unlock_t {
    void*    handle;
};

//-----------------------------------------------------------------------------

struct gpudma_state_t {
    void*       handle;
    size_t      page_count;
    size_t      page_size;
    uint64_t    pages[1];
};

//-----------------------------------------------------------------------------


#endif //_GPUDMAIOTCL_H_
