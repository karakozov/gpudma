
#ifndef __GPUDMA_H__
#define __GPUDMA_H__

#include <linux/cdev.h>
#include <linux/sched.h>
#include <linux/version.h>
#include <linux/semaphore.h>

#ifndef __GPUDMAIOTCL_H__
    #include "gpudmaioctl.h"
#endif

//-----------------------------------------------------------------------------

#define ms_to_jiffies( ms ) (HZ*ms/1000)
#define jiffies_to_ms( jf ) (jf*1000/HZ)

//-----------------------------------------------------------------------------

#define MEM_ID    0xBABECAFE

struct gpudma_mem_t {

    struct list_head        mem_list;
    char                    mem_name[128];
    void*                   mem_file;
    void*                   mem_handle;
    atomic_t                mem_owner_count;
    atomic_t                mem_lock_count;
    atomic_t                mem_unlock_count;
    struct semaphore        mem_sem;
    u32                     mem_id;
};

//-----------------------------------------------------------------------------

struct gpudma_driver {

    dev_t                   m_devno;
    struct list_head        m_list;
    char                    m_name[128];
    struct mutex            m_mutex;
    struct semaphore        m_sem;
    spinlock_t              m_lock;
    atomic_t                m_usage;
    int                     m_index;

    struct cdev             m_cdev;
    struct proc_dir_entry*  m_proc;

    struct list_head        m_mem_list;
    struct mutex            m_mem_lock;

};

//-----------------------------------------------------------------------------

void* gpudma_mem_create( struct gpudma_driver *drv, struct gpudma_create_t *param );
int gpudma_mem_lock( struct gpudma_driver *drv, struct gpudma_lock_t *param );
int gpudma_mem_unlock( struct gpudma_driver *drv, struct gpudma_unlock_t *param );
int gpudma_mem_close( struct gpudma_driver *drv, struct gpudma_close_t *param );
int gpudma_mem_close_all( struct gpudma_driver *drv );

//-----------------------------------------------------------------------------

#endif //__GPUDMA_H__
