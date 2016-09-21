
#ifndef _IOCTLRW_H_
#define _IOCTLRW_H_

//-----------------------------------------------------------------------------

int ioctl_mem_open(struct gpudma_driver *drv, unsigned long arg);
int ioctl_mem_close(struct gpudma_driver *drv, unsigned long arg);
int ioctl_mem_lock(struct gpudma_driver *drv, unsigned long arg);
int ioctl_mem_unlock(struct gpudma_driver *drv, unsigned long arg);

//-----------------------------------------------------------------------------

#endif //_IOCTLRW_H_
