

#ifndef GPUMEM_H
#define GPUMEM_H

//-----------------------------------------------------------------------------

#include <linux/cdev.h>
#include <linux/sched.h>
#include <linux/version.h>
#include <linux/semaphore.h>

#include "nv-p2p.h"

//-----------------------------------------------------------------------------

struct gpumem {
    struct semaphore         sem;
    struct proc_dir_entry*   proc;
    nvidia_p2p_page_table_t* page_table;
    u64 virt_start;
};

//-----------------------------------------------------------------------------

int get_nv_page_size(int val);

//-----------------------------------------------------------------------------

#endif
