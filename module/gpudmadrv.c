
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
#include <asm/io.h>

#include <asm/uaccess.h>
#include <linux/types.h>
#include <linux/ioport.h>
#include <linux/poll.h>
#include <linux/slab.h>
#include <linux/interrupt.h>

#include "gpudmadrv.h"
#include "gpudmaioctl.h"
#include "ioctlrw.h"
#include "gpudmaproc.h"

//-----------------------------------------------------------------------------

MODULE_AUTHOR("Vladimir Karakozov. karakozov@gmail.com");
MODULE_LICENSE("GPL");

//-----------------------------------------------------------------------------

static dev_t devno = MKDEV(0, 0);
static LIST_HEAD(gpudma_list);
static struct mutex gpudma_mutex;
static struct class  *gpudma_class = 0;
static struct device *gpudma_device;

//-----------------------------------------------------------------------------

static struct gpudma_driver *file_to_device( struct file *file )
{
    return (struct gpudma_driver*)file->private_data;
}

//--------------------------------------------------------------------

static int gpudma_device_fasync(int fd, struct file *file, int mode)
{
    struct gpudma_driver *pDriver = file->private_data;

    printk(KERN_DEBUG"%s()\n", __FUNCTION__);

    if(!pDriver)
        return -ENODEV;

    return 0;
}

//-----------------------------------------------------------------------------

static unsigned int gpudma_device_poll(struct file *filp, poll_table *wait)
{
    unsigned int mask = 0;
    struct gpudma_driver *pDriver = file_to_device(filp);

    printk(KERN_DEBUG"%s()\n", __FUNCTION__);

    if(!pDriver)
        return -ENODEV;

    return mask;
}

//-----------------------------------------------------------------------------

static int gpudma_device_open( struct inode *inode, struct file *file )
{
    struct gpudma_driver *pDriver = container_of(inode->i_cdev, struct gpudma_driver, m_cdev);
    if(!pDriver) {
        printk(KERN_DEBUG"%s(): Open driver failed\n", __FUNCTION__);
        return -ENODEV;
    }

    mutex_lock(&pDriver->m_mutex);

    atomic_inc(&pDriver->m_usage);

    file->private_data = (void*)pDriver;

    printk(KERN_DEBUG"%s(): Open driver %s. m_usage = %d. file = %p\n", __FUNCTION__, pDriver->m_name, atomic_read(&pDriver->m_usage), file);

    mutex_unlock(&pDriver->m_mutex);

    return 0;
}

//-----------------------------------------------------------------------------

static int gpudma_device_close( struct inode *inode, struct file *file )
{
    struct gpudma_driver *pDriver = container_of(inode->i_cdev, struct gpudma_driver, m_cdev);
    if(!pDriver) {
        printk(KERN_DEBUG"%s(): Close driver failed\n", __FUNCTION__);
        return -ENODEV;
    }

    mutex_lock(&pDriver->m_mutex);

    atomic_dec(&pDriver->m_usage);

    file->private_data = NULL;

    printk(KERN_DEBUG"%s(): Close driver %s. m_usage = %d. file = %p\n", __FUNCTION__, pDriver->m_name, atomic_read(&pDriver->m_usage), file);

    if(atomic_read(&pDriver->m_usage) == 0) {
        gpudma_mem_close_all(pDriver);
    }

    mutex_unlock(&pDriver->m_mutex);

    return 0;
}

//-----------------------------------------------------------------------------

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,34)
static long gpudma_device_ioctl( struct file *file, unsigned int cmd, unsigned long arg )
#else
static int gpudma_device_ioctl( struct inode *inode, struct file *file, unsigned int cmd, unsigned long arg )
#endif
{
    int error = 0;
    struct gpudma_driver *pDriver = file_to_device(file);
    if(!pDriver) {
        printk(KERN_DEBUG"%s(): ioctl driver failed\n", __FUNCTION__);
        return -ENODEV;
    }

    printk(KERN_DEBUG"%s()\n", __FUNCTION__ );

    switch(cmd) {

    case IOCTL_GPUDMA_MEM_OPEN:
        error = ioctl_mem_open(pDriver, arg);
        break;
    case IOCTL_GPUDMA_MEM_LOCK:
        error = ioctl_mem_lock(pDriver, arg);
        break;
    case IOCTL_GPUDMA_MEM_UNLOCK:
        error = ioctl_mem_unlock(pDriver, arg);
        break;
    case IOCTL_GPUDMA_MEM_CLOSE:
        error = ioctl_mem_close(pDriver, arg);
        break;
    default:
        printk(KERN_DEBUG"%s(): Unknown ioctl command\n", __FUNCTION__);
        error = -EINVAL;
        break;
    }

    return error;
}

//-----------------------------------------------------------------------------

struct file_operations gpudma_fops = {

    .owner = THIS_MODULE,
    .read = NULL,
    .write = NULL,

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,34)
    .unlocked_ioctl = gpudma_device_ioctl,
    .compat_ioctl = gpudma_device_ioctl,
#else
    .ioctl = gpudma_device_ioctl,
#endif

    .mmap = NULL,
    .open = gpudma_device_open,
    .release = gpudma_device_close,
    .fasync = gpudma_device_fasync,
    .poll = gpudma_device_poll,
};

//-----------------------------------------------------------------------------

static int  __init gpudma_device_probe(void)
{
    int error = 0;
    struct gpudma_driver *drv = NULL;

    printk(KERN_DEBUG"%s()\n", __FUNCTION__);

    drv = kzalloc(sizeof(struct gpudma_driver), GFP_KERNEL);
    if(!drv) {
        error = -ENOMEM;
        goto do_out;
    }

    INIT_LIST_HEAD(&drv->m_list);

    mutex_init(&drv->m_mutex);
    sema_init(&drv->m_sem, 1);
    spin_lock_init(&drv->m_lock);
    atomic_set(&drv->m_usage, 0);
    drv->m_index = 0;

    INIT_LIST_HEAD(&drv->m_mem_list);
    mutex_init(&drv->m_mem_lock);

    cdev_init(&drv->m_cdev, &gpudma_fops);
    drv->m_cdev.owner = THIS_MODULE;
    drv->m_cdev.ops = &gpudma_fops;
    drv->m_devno = MKDEV(MAJOR(devno), 0);

    snprintf(drv->m_name, sizeof(drv->m_name), "%s", GPUDMA_DRIVER_NAME);

    error = cdev_add(&drv->m_cdev, drv->m_devno, 1);
    if(error) {
        printk(KERN_DEBUG"%s(): Error add char device %d\n", __FUNCTION__, 0);
        error = -EINVAL;
        goto do_free_memory;
    }

    printk(KERN_DEBUG"%s(): Add cdev %d\n", __FUNCTION__, 0);

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,34)
    gpudma_device = device_create(gpudma_class, NULL, drv->m_devno, "%s", drv->m_name);
#else
    gpudma_device = device_create(gpudma_class, NULL, drv->m_devno, NULL, "%s", drv->m_name);
#endif
    if(!gpudma_device) {
        printk(KERN_DEBUG"%s(): Error create device for board: %s\n", __FUNCTION__, drv->m_name);
        error = -EINVAL;
        goto do_delete_cdev;
    }

    printk(KERN_DEBUG"%s(): Create device file for board: %s\n", __FUNCTION__, drv->m_name);

    gpudma_register_proc(drv->m_name, NULL, drv);

    printk(KERN_DEBUG"%s(): Driver %s - setup complete\n", __FUNCTION__, drv->m_name);

    list_add_tail(&drv->m_list, &gpudma_list);

    return error;

do_delete_cdev:
    cdev_del(&drv->m_cdev);

do_free_memory:
    kfree(drv);

do_out:
    return error;
}

//-----------------------------------------------------------------------------

static void __exit gpudma_device_remove(void)
{
    struct list_head *pos, *n;
    struct gpudma_driver *entry = NULL;

    printk(KERN_DEBUG"%s()\n", __FUNCTION__);

    list_for_each_safe(pos, n, &gpudma_list) {

        entry = list_entry(pos, struct gpudma_driver, m_list);

        gpudma_mem_close_all(entry);
        gpudma_remove_proc(entry->m_name);
        device_destroy(gpudma_class, entry->m_devno);
        cdev_del(&entry->m_cdev);
        list_del(pos);
        kfree(entry);
    }
}

//-----------------------------------------------------------------------------

static int __init gpudma_module_init(void)
{
    int error = 0;

    printk(KERN_DEBUG"%s()\n", __FUNCTION__);

    mutex_init(&gpudma_mutex);
    mutex_lock(&gpudma_mutex);

    error = alloc_chrdev_region(&devno, 0, MAX_GPUDMA_DEVICE_SUPPORT, GPUDMA_DRIVER_NAME);
    if(error < 0) {
        printk(KERN_DEBUG"%s(): Erorr allocate char device regions\n", __FUNCTION__);
        goto do_out;
    }

    printk(KERN_DEBUG"%s(): Allocate %d device numbers. Major number = %d\n", __FUNCTION__, MAX_GPUDMA_DEVICE_SUPPORT, MAJOR(devno));

    gpudma_class = class_create(THIS_MODULE, GPUDMA_DRIVER_NAME);
    if(!gpudma_class) {
        printk(KERN_DEBUG"%s(): Erorr create GPUDMA device class: %s\n", __FUNCTION__, GPUDMA_DRIVER_NAME);
        error = -EINVAL;
        goto do_free_chrdev;
    }

    error = gpudma_device_probe();
    if(error < 0) {
        printk(KERN_DEBUG"%s(): Erorr probe GPUDMA driver\n", __FUNCTION__);
        error = -EINVAL;
        goto do_delete_class;
    }

    mutex_unlock(&gpudma_mutex);

    return 0;

do_delete_class:
    class_destroy(gpudma_class);

do_free_chrdev:
    unregister_chrdev_region(devno, MAX_GPUDMA_DEVICE_SUPPORT);

do_out:
    mutex_unlock(&gpudma_mutex);

    return error;
}

//-----------------------------------------------------------------------------

static void __exit gpudma_module_cleanup(void)
{
    printk(KERN_DEBUG"%s()\n", __FUNCTION__);

    mutex_lock(&gpudma_mutex);

    gpudma_device_remove();

    if(gpudma_class)
        class_destroy(gpudma_class);

    unregister_chrdev_region(devno, MAX_GPUDMA_DEVICE_SUPPORT);

    mutex_unlock(&gpudma_mutex);
}

//-----------------------------------------------------------------------------

module_init(gpudma_module_init);
module_exit(gpudma_module_cleanup);

//-----------------------------------------------------------------------------
