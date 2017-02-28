

#ifndef GPUMEM_H
#define GPUMEM_H

//-----------------------------------------------------------------------------

#include <linux/cdev.h>
#include <linux/sched.h>
#include <linux/version.h>
#include <linux/semaphore.h>

#include "nv-p2p.h"

//-----------------------------------------------------------------------------

struct gpumem_t {
    struct list_head list;
    void *handle;
    u64 virt_start;
    nvidia_p2p_page_table_t* page_table;
};

//-----------------------------------------------------------------------------

struct gpumem {
    struct semaphore         sem;
    struct proc_dir_entry*   proc;
    struct list_head         table_list;
};

//-----------------------------------------------------------------------------

int get_nv_page_size(int val);

//-----------------------------------------------------------------------------

#endif
