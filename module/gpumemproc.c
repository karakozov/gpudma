
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
#include <linux/semaphore.h>
#include <asm/io.h>

#include "gpumemdrv.h"
#include "gpumemproc.h"

//--------------------------------------------------------------------

struct log_buf_t {
    struct seq_file *param;
};

//--------------------------------------------------------------------

#define print_info(S...) seq_printf(S)

//--------------------------------------------------------------------

static void show_mem_info( struct gpumem *drv, struct seq_file *m )
{
    struct list_head *pos, *n;
    int i=0, idx=0;
    if(!drv || !m) {
        printk(KERN_DEBUG"%s(): EINVAL\n", __FUNCTION__ );
        return;
    }

    print_info(m, "%s\n", "Pinned memory info:");

    list_for_each_safe(pos, n, &drv->table_list) {

        struct gpumem_t *entry = list_entry(pos, struct gpumem_t, list);
        if(entry) {
            if(entry->virt_start) {

                print_info(m, "%d: Entry - %p\n", idx, entry);
                print_info(m, "Virtual GPU address - 0x%llx\n", entry->virt_start);
                print_info(m, "Number of pages - %d\n", entry->page_table->entries);
                print_info(m, "Page size - 0x%x\n", get_nv_page_size(entry->page_table->page_size));

                for(i=0; i<entry->page_table->entries; i++) {
                    struct nvidia_p2p_page *nvp = entry->page_table->pages[i];
                    if(nvp) {
                        print_info(m, "%02d: - 0x%llx\n", i, nvp->physical_address);
                    }
                }

                print_info(m, "\n");

                ++idx;
            }
        }
    }
}

//--------------------------------------------------------------------

static int gpumem_proc_show(struct seq_file *m, void *v)
{
    struct gpumem *p = m->private;

    show_mem_info( p, m );

    return 0;
}

//--------------------------------------------------------------------

static int gpumem_proc_open(struct inode *inode, struct file *file)
{
#if (LINUX_VERSION_CODE > KERNEL_VERSION(3, 9, 0))
    struct gpumem *p = (struct gpumem *)PDE_DATA(inode);
#else
    struct gpumem *p = (struct gpumem *)PDE(inode)->data;
#endif
    return single_open(file, gpumem_proc_show, p);
}

//--------------------------------------------------------------------

static int gpumem_proc_release(struct inode *inode, struct file *file)
{
    return single_release(inode, file);
}

//--------------------------------------------------------------------

static const struct file_operations gpumem_proc_fops = {
    .owner          = THIS_MODULE,
    .open           = gpumem_proc_open,
    .read           = seq_read,
    .llseek         = seq_lseek,
    .release        = gpumem_proc_release,
};

//--------------------------------------------------------------------

void gpumem_register_proc( char *name, void *fptr, void *data )
{
    struct gpumem *p = (struct gpumem*)data;

    if(!data) {
        printk(KERN_DEBUG"%s(): Invalid driver pointer\n", __FUNCTION__ );
        return;
    }

    p->proc = proc_create_data(name, S_IRUGO, NULL, &gpumem_proc_fops, p);
    if(!p->proc) {
        printk(KERN_DEBUG"%s(): Error register /proc entry\n", __FUNCTION__);
    }
}

//--------------------------------------------------------------------

void gpumem_remove_proc( char *name )
{
    remove_proc_entry(name, NULL);
}

//--------------------------------------------------------------------

