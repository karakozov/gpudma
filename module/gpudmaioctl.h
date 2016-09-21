
#ifndef __GPUDMAIOTCL_H__
#define __GPUDMAIOTCL_H__

//-----------------------------------------------------------------------------

#define GPUDMA_DRIVER_NAME             "gpudma"
#define MAX_GPUDMA_DEVICE_SUPPORT       1

//-----------------------------------------------------------------------------

#ifdef __linux__
#include <linux/types.h>
#ifndef __KERNEL__
#include <sys/ioctl.h>
#endif
#define GPUDMA_DEVICE_TYPE             'i'
#define GPUDMA_MAKE_IOCTL(c) _IO(GPUDMA_DEVICE_TYPE, (c))
#endif

#define IOCTL_GPUDMA_MEM_OPEN		GPUDMA_MAKE_IOCTL(10)
#define IOCTL_GPUDMA_MEM_LOCK		GPUDMA_MAKE_IOCTL(11)
#define IOCTL_GPUDMA_MEM_UNLOCK		GPUDMA_MAKE_IOCTL(12)
#define IOCTL_GPUDMA_MEM_CLOSE		GPUDMA_MAKE_IOCTL(13)

//-----------------------------------------------------------------------------

struct gpudma_create_t {

    char    name[128];
    int     value;
    int     flag;
    void    *handle;
};

//-----------------------------------------------------------------------------

struct gpudma_lock_t {

    void    *handle;
};

//-----------------------------------------------------------------------------

struct gpudma_unlock_t {

    void    *handle;
};

//-----------------------------------------------------------------------------

struct gpudma_reset_t {

    void    *handle;
};

//-----------------------------------------------------------------------------

struct gpudma_close_t {

    void    *handle;
};

//-----------------------------------------------------------------------------

#endif //_GPUDMAIOTCL_H_
