
#include <linux/kernel.h>
#define __NO_VERSION__
#include <linux/module.h>
#include <linux/types.h>
#include <linux/version.h>
#include <linux/ioport.h>
#include <linux/pci.h>
#include <linux/pagemap.h>
#include <linux/interrupt.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <asm/io.h>

#include "gpudmadrv.h"
#include "gpudmaproc.h"

//--------------------------------------------------------------------

struct log_buf_t {
    struct seq_file *param;
};

//--------------------------------------------------------------------

#define print_info(S...) seq_printf(S)

//--------------------------------------------------------------------

static int show_mem_info( struct gpudma_driver *drv, struct seq_file *m )
{
    struct list_head *pos, *n;
    struct gpudma_mem_t *entry = NULL;
    int mem_counter = 0;

    if(!drv || !m) {
        printk(KERN_DEBUG"%s(): EINVAL\n", __FUNCTION__ );
        return -1;
    }

    seq_printf(m, "%s\n", "Memory addresses");

    mutex_lock(&drv->m_mem_lock);

    list_for_each_safe(pos, n, &drv->m_mem_list) {

        entry = list_entry(pos, struct gpudma_mem_t, mem_list);

        if(entry) {

            seq_printf(m, "%d: %s (lock %d, unlock %d, usage %d)\n", mem_counter,
                       entry->mem_name,
                       atomic_read(&entry->mem_lock_count),
                       atomic_read(&entry->mem_unlock_count),
                       atomic_read(&entry->mem_owner_count));
            mem_counter++;
        }
    }

    mutex_unlock(&drv->m_mem_lock);

    seq_printf(m, "Total memory: %d\n", mem_counter );

    return mem_counter;
}

//--------------------------------------------------------------------

static int gpudma_proc_show(struct seq_file *m, void *v)
{
    struct gpudma_driver *p = m->private;

    show_mem_info( p, m );

    return 0;
}

//--------------------------------------------------------------------

static int gpudma_proc_open(struct inode *inode, struct file *file)
{
#if LINUX_VERSION_CODE >= KERNEL_VERSION(3,10,0)
    struct gpudma_driver *p = (struct gpudma_driver *)PDE_DATA(inode);
#else
    struct gpudma_driver *p = PDE(inode)->data;
#endif

    return single_open(file, gpudma_proc_show, p);
}

//--------------------------------------------------------------------

static int gpudma_proc_release(struct inode *inode, struct file *file)
{
    return single_release(inode, file);
}

//--------------------------------------------------------------------

static const struct file_operations gpudma_proc_fops = {
    .owner          = THIS_MODULE,
    .open           = gpudma_proc_open,
    .read           = seq_read,
    .llseek         = seq_lseek,
    .release        = gpudma_proc_release,
};

//--------------------------------------------------------------------

void gpudma_register_proc( char *name, void *fptr, void *data )
{
    struct gpudma_driver *p = (struct gpudma_driver*)data;

    if(!data) {
        printk(KERN_DEBUG"%s(): Invalid driver pointer\n", __FUNCTION__ );
        return;
    }

    p->m_proc = proc_create_data(name, S_IRUGO, NULL, &gpudma_proc_fops, p);
    if(!p->m_proc) {
        printk(KERN_DEBUG"%s(): Error register /proc entry\n", __FUNCTION__);
    }
}

//--------------------------------------------------------------------

void gpudma_remove_proc( char *name )
{
    remove_proc_entry(name, NULL);
}

//--------------------------------------------------------------------

