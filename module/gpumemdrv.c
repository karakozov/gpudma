
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/version.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/ioport.h>
#include <linux/list.h>
#include <linux/pci.h>
#include <linux/proc_fs.h>
#include <linux/interrupt.h>
#include <linux/miscdevice.h>
#include <linux/platform_device.h>
//#include <linux/of.h>
//#include <linux/of_platform.h>
#include <asm/io.h>

#include <asm/uaccess.h>
#include <linux/types.h>
#include <linux/ioport.h>
#include <linux/poll.h>
#include <linux/slab.h>
#include <linux/interrupt.h>

#include "gpumemdrv.h"
#include "ioctlrw.h"
#include "gpumemioctl.h"
#include "gpumemproc.h"

//-----------------------------------------------------------------------------

MODULE_AUTHOR("Vladimir Karakozov. karakozov@gmail.com");
MODULE_LICENSE("GPL");

//-----------------------------------------------------------------------------
static struct gpumem dev;
//-----------------------------------------------------------------------------

static struct gpumem *file_to_device( struct file *file )
{
    return (struct gpumem*)file->private_data;
}

//--------------------------------------------------------------------

static int gpumem_open( struct inode *inode, struct file *file )
{
    file->private_data = (void*)&dev;
    return 0;
}

//-----------------------------------------------------------------------------

static int gpumem_close( struct inode *inode, struct file *file )
{
    file->private_data = 0;
    return 0;
}

//-----------------------------------------------------------------------------

static long gpumem_ioctl( struct file *file, unsigned int cmd, unsigned long arg )
{
    int error = 0;
    struct gpumem *dev = file_to_device(file);
    if(!dev) {
        printk(KERN_ERR"%s(): ioctl driver failed\n", __FUNCTION__);
        return -ENODEV;
    }

    switch(cmd) {

    case IOCTL_GPUMEM_LOCK: error = ioctl_mem_lock(dev, arg); break;
    case IOCTL_GPUMEM_UNLOCK: error = ioctl_mem_unlock(dev, arg); break;
    case IOCTL_GPUMEM_STATE: error = ioctl_mem_state(dev, arg); break;
    default:
        printk(KERN_DEBUG"%s(): Unknown ioctl command\n", __FUNCTION__);
        error = -EINVAL;
        break;
    }

    return error;
}

//-----------------------------------------------------------------------------

int gpumem_mmap(struct file *file, struct vm_area_struct *vma)
{
    size_t size = vma->vm_end - vma->vm_start;

    if (!(vma->vm_flags & VM_MAYSHARE))
        return -EINVAL;

    vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);

    if (remap_pfn_range(vma,
                        vma->vm_start,
                        vma->vm_pgoff,
                        size,
                        vma->vm_page_prot)) {
        pr_err("%s(): error in remap_page_range.\n", __func__ );
        return -EAGAIN;
    }

    return 0;
}

//-----------------------------------------------------------------------------

struct file_operations gpumem_fops = {

    .owner = THIS_MODULE,
    .unlocked_ioctl = gpumem_ioctl,
    .compat_ioctl = gpumem_ioctl,
    .open = gpumem_open,
    .release = gpumem_close,
    .mmap = gpumem_mmap,
};

//-----------------------------------------------------------------------------

static struct miscdevice gpumem_dev = {

    MISC_DYNAMIC_MINOR,
    GPUMEM_DRIVER_NAME,
    &gpumem_fops
};

//-----------------------------------------------------------------------------

static int __init gpumem_init(void)
{
    pr_info(GPUMEM_DRIVER_NAME ": %s()\n", __func__);
    dev.proc = 0;
    sema_init(&dev.sem, 1);
    INIT_LIST_HEAD(&dev.table_list);
    gpumem_register_proc(GPUMEM_DRIVER_NAME, 0, &dev);
    misc_register(&gpumem_dev);
    return 0;
}

//-----------------------------------------------------------------------------

static void __exit gpumem_cleanup(void)
{
    pr_info(GPUMEM_DRIVER_NAME ": %s()\n", __func__);
    gpumem_remove_proc(GPUMEM_DRIVER_NAME);
    misc_deregister(&gpumem_dev);
}

//-----------------------------------------------------------------------------

module_init(gpumem_init);
module_exit(gpumem_cleanup);

//-----------------------------------------------------------------------------
