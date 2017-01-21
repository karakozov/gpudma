
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

#include "gpumemdrv.h"
#include "gpumemioctl.h"

//-----------------------------------------------------------------------------

int get_nv_page_size(int val)
{
    switch(val) {
    case NVIDIA_P2P_PAGE_SIZE_4KB: return 4*1024;
    case NVIDIA_P2P_PAGE_SIZE_64KB: return 64*1024;
    case NVIDIA_P2P_PAGE_SIZE_128KB: return 128*1024;
    }
    return 0;
}

//--------------------------------------------------------------------

void free_nvp_callback(void *data)
{
    struct gpumem *drv = (struct gpumem *)data;
    int res;
    res = nvidia_p2p_free_page_table(drv->page_table);
    if(res == 0) {
        printk(KERN_ERR"%s(): nvidia_p2p_free_page_table() - OK!\n", __FUNCTION__);
        drv->virt_start = 0ULL;
        drv->page_table = 0;
    } else {
        printk(KERN_ERR"%s(): Error in nvidia_p2p_free_page_table()\n", __FUNCTION__);
    }
}

//-----------------------------------------------------------------------------

int ioctl_mem_lock(struct gpumem *drv, unsigned long arg)
{
    int error = 0;
    size_t pin_size = 0ULL;
    struct gpudma_lock_t param;

    if(copy_from_user(&param, (void *)arg, sizeof(struct gpudma_lock_t))) {
        printk(KERN_ERR"%s(): Error in copy_from_user()\n", __FUNCTION__);
        error = -EFAULT;
        goto do_exit;
    }

    if(drv->virt_start != 0ULL) {
        printk(KERN_ERR"%s(): Error memory already pinned!\n", __FUNCTION__);
        return -EINVAL;
    }

    drv->virt_start = (param.addr & GPU_BOUND_MASK);
    pin_size = (param.addr + param.size - drv->virt_start);
    if(!pin_size) {
        printk(KERN_ERR"%s(): Error invalid memory size!\n", __FUNCTION__);
        return -EINVAL;
    }

    error = nvidia_p2p_get_pages(0, 0, drv->virt_start, pin_size, &drv->page_table, free_nvp_callback, drv);
    if(error != 0) {
        printk(KERN_ERR"%s(): Error in nvidia_p2p_get_pages()\n", __FUNCTION__);
        return error;
    }

    param.page_count = drv->page_table->entries;

    if(copy_to_user((void *)arg, &param, sizeof(struct gpudma_lock_t))) {
        printk(KERN_ERR"%s(): Error in copy_from_user()\n", __FUNCTION__);
        error = -EFAULT;
        goto do_exit;
    }

do_exit:
    return error;
}

//-----------------------------------------------------------------------------

int ioctl_mem_unlock(struct gpumem *drv, unsigned long arg)
{
    int error = -EINVAL;

    if(drv->virt_start) {
        error = nvidia_p2p_put_pages(0, 0, drv->virt_start, drv->page_table);
        if(error != 0) {
            printk(KERN_ERR"%s(): Error in nvidia_p2p_put_pages()\n", __FUNCTION__);
            goto do_exit;
        }
        drv->virt_start = 0ULL;
        printk(KERN_ERR"%s(): nvidia_p2p_put_pages() - Ok!\n", __FUNCTION__);
    }

do_exit:
    return error;
}

//-----------------------------------------------------------------------------

int ioctl_mem_state(struct gpumem *drv, unsigned long arg)
{
    int error = 0;
    int size = 0;
    size_t i=0;
    struct gpudma_state_t header;
    struct gpudma_state_t *param;

    if(copy_from_user(&header, (void *)arg, sizeof(struct gpudma_state_t))) {
        printk(KERN_ERR"%s(): Error in copy_from_user()\n", __FUNCTION__);
        error = -EFAULT;
        goto do_exit;
    }

    if(!drv->page_table) {
        printk(KERN_ERR"%s(): Error - memory not pinned!\n", __FUNCTION__);
        return -EINVAL;
    }

    if(drv->page_table->entries != header.page_count) {
        printk(KERN_ERR"%s(): Error - page counters invalid!\n", __FUNCTION__);
        return -EINVAL;
    }

    size = (sizeof(uint64_t)*header.page_count) + sizeof(struct gpudma_state_t);
    param = kzalloc(size, GFP_KERNEL);
    if(!param) {
        printk(KERN_ERR"%s(): Error allocate memory!\n", __FUNCTION__);
        return -ENOMEM;
    }
    param->page_size = get_nv_page_size(drv->page_table->page_size);
    for(i=0; i<drv->page_table->entries; i++) {
        struct nvidia_p2p_page *nvp = drv->page_table->pages[i];
        if(nvp) {
            param->pages[i] = nvp->physical_address;
            param->page_count++;
            printk(KERN_ERR"%s(): %02ld - 0x%llx\n", __FUNCTION__, i, param->pages[i]);
        }
    }
    printk(KERN_ERR"%s(): page_count = %ld\n", __FUNCTION__, param->page_count);

    if(copy_to_user((void *)arg, param, size)) {
        printk(KERN_DEBUG"%s(): Error in copy_to_user()\n", __FUNCTION__);
        error = -EFAULT;
    }

    kfree(param);

do_exit:
    return error;
}

//-----------------------------------------------------------------------------
