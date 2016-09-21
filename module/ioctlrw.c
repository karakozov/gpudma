
#include <linux/kernel.h>
#define __NO_VERSION__
#include <linux/module.h>
#include <linux/types.h>
#include <linux/ioport.h>
#include <linux/pci.h>
#include <linux/pagemap.h>
#include <linux/interrupt.h>
#include <linux/proc_fs.h>
#include <asm/io.h>

#include "gpudmadrv.h"

//-----------------------------------------------------------------------------

int ioctl_mem_open(struct gpudma_driver *drv, unsigned long arg)
{
    int error = 0;
    struct gpudma_create_t param;

    if(copy_from_user(&param, (void *)arg, sizeof(struct gpudma_create_t))) {
        printk(KERN_DEBUG"%s(): Error in copy_from_user()\n", __FUNCTION__);
        error = -EFAULT;
        goto do_exit;
    }

    param.handle = gpudma_mem_create( drv, &param );
    if(!param.handle) {
        printk(KERN_DEBUG"%s(): Error in gpudma_mem_create()\n", __FUNCTION__);
        error = -EINVAL;
        goto do_exit;
    }

    if(copy_to_user((void*)arg, (void*)&param, sizeof(struct gpudma_create_t))) {
        printk(KERN_DEBUG"%s(): Error in copy_to_user()\n", __FUNCTION__);
        error = -EFAULT;
        goto do_exit;
    }

do_exit:
    return error;
}

//-----------------------------------------------------------------------------

int ioctl_mem_close(struct gpudma_driver *drv, unsigned long arg)
{
    int error = 0;
    struct gpudma_close_t param;

    if(copy_from_user(&param, (void *)arg, sizeof(struct gpudma_close_t))) {
        printk(KERN_DEBUG"%s(): Error in copy_from_user()\n", __FUNCTION__);
        error = -EFAULT;
        goto do_exit;
    }

    error = gpudma_mem_close( drv, &param );
    if(error < 0) {
        printk(KERN_DEBUG"%s(): Error in gpudma_mem_close()\n", __FUNCTION__);
        goto do_exit;
    }

do_exit:
    return error;
}

//-----------------------------------------------------------------------------

int ioctl_mem_lock(struct gpudma_driver *drv, unsigned long arg)
{
    int error = 0;
    struct gpudma_lock_t param;

    if(copy_from_user(&param, (void *)arg, sizeof(struct gpudma_lock_t))) {
        printk(KERN_DEBUG"%s(): Error in copy_from_user()\n", __FUNCTION__);
        error = -EFAULT;
        goto do_exit;
    }

    error = gpudma_mem_lock( drv, &param );
    if(error < 0) {
        printk(KERN_DEBUG"%s(): Error in gpudma_mem_lock()\n", __FUNCTION__);
        goto do_exit;
    }

do_exit:
    return error;
}

//-----------------------------------------------------------------------------

int ioctl_mem_unlock(struct gpudma_driver *drv, unsigned long arg)
{
    int error = 0;
    struct gpudma_unlock_t param;

    if(copy_from_user(&param, (void *)arg, sizeof(struct gpudma_unlock_t))) {
        printk(KERN_DEBUG"%s(): Error in copy_from_user()\n", __FUNCTION__);
        error = -EFAULT;
        goto do_exit;
    }

    error = gpudma_mem_unlock( drv, &param );
    if(error < 0) {
        printk(KERN_DEBUG"%s(): Error in gpudma_mem_unlock()\n", __FUNCTION__);
        goto do_exit;
    }

do_exit:
    return error;
}

//-----------------------------------------------------------------------------
