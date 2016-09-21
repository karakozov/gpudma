
#include <linux/kernel.h>
#define __NO_VERSION__
#include <linux/module.h>
#include <linux/types.h>
#include <linux/ioport.h>
#include <linux/pci.h>
#include <linux/pagemap.h>
#include <linux/interrupt.h>
#include <linux/proc_fs.h>
#include <linux/slab.h>
#include <asm/io.h>

#include "gpudmadrv.h"
#include "gpudmaioctl.h"

//-----------------------------------------------------------------------------

void* gpudma_mem_create( struct gpudma_driver *drv, struct gpudma_create_t *param )
{
    bool exist = false;
    struct list_head *pos, *n;
    struct gpudma_mem_t *entry = NULL;
    struct gpudma_mem_t *mem = NULL;

    printk(KERN_DEBUG"%s()\n", __FUNCTION__ );

    if(!drv || !param) {
        printk(KERN_DEBUG"%s(): Invalid parameters\n", __FUNCTION__ );
        goto do_out;
    }

    printk(KERN_DEBUG"%s(): name = %s\n", __FUNCTION__, param->name );
    printk(KERN_DEBUG"%s(): value = %d\n", __FUNCTION__, param->value );

    mutex_lock(&drv->m_mem_lock);

    list_for_each_safe(pos, n, &drv->m_mem_list) {

        entry = list_entry(pos, struct gpudma_mem_t, mem_list);

        if(strcmp(entry->mem_name, param->name) == 0) {
            mem = entry;
            exist = true;
            break;
        }
    }

    if(!exist) {

        printk(KERN_DEBUG"%s(): memory name = %s was not found. Create new memory object\n", __FUNCTION__, param->name );

        mem = (struct gpudma_mem_t*)kzalloc(sizeof(struct gpudma_mem_t), GFP_KERNEL);
        if(!mem) {
            printk(KERN_DEBUG"%s(): Error allocate memory to object\n", __FUNCTION__ );
            goto do_out;
        }

        INIT_LIST_HEAD(&mem->mem_list);
        sema_init(&mem->mem_sem, param->value);
        snprintf(mem->mem_name, sizeof(mem->mem_name), "%s", param->name);
        mem->mem_handle = mem;
        mem->mem_id = MEM_ID;
        atomic_set(&mem->mem_owner_count, 0);

        list_add_tail(&mem->mem_list, &drv->m_mem_list);

    } else {

        printk(KERN_DEBUG"%s(): Memory name = %s was found. Use exist object\n", __FUNCTION__, param->name );
    }

    atomic_inc(&mem->mem_owner_count);
    param->handle = mem->mem_handle;

    printk(KERN_DEBUG"%s(): %s - mem_owner_count: %d\n", __FUNCTION__, param->name, atomic_read(&mem->mem_owner_count) );

do_out:
    mutex_unlock(&drv->m_mem_lock);

    return &mem->mem_handle;
}

//-----------------------------------------------------------------------------

int gpudma_mem_lock( struct gpudma_driver *drv, struct gpudma_lock_t *param )
{
    bool exist = false;
    int error = -EINVAL;
    struct gpudma_mem_t *entry = NULL;

    printk(KERN_DEBUG"%s()\n", __FUNCTION__ );

    if(!drv || !param || !param->handle) {
        printk(KERN_DEBUG"%s(): Invalid parameters\n", __FUNCTION__ );
        goto do_out;
    }

    mutex_lock(&drv->m_mem_lock);

    entry = container_of(param->handle, struct gpudma_mem_t, mem_handle);
    if(entry && (entry->mem_id == MEM_ID)) {
        exist = true;
    }

    mutex_unlock(&drv->m_mem_lock);

    if(exist) {

        printk(KERN_DEBUG"%s(): %s - mem_owner_count: %d\n", __FUNCTION__, entry->mem_name, atomic_read(&entry->mem_owner_count) );

        //TODO: nvidia_p2p_get_pages()
        atomic_inc(&entry->mem_lock_count);

        printk(KERN_DEBUG"%s(): %s - locked %d\n", __FUNCTION__, entry->mem_name, atomic_read(&entry->mem_lock_count) );

    } else {

        printk(KERN_DEBUG"%s(): Invalid handle\n", __FUNCTION__ );
    }

do_out:
    return error;
}

//-----------------------------------------------------------------------------

int gpudma_mem_unlock( struct gpudma_driver *drv, struct gpudma_unlock_t *param )
{
    bool exist = false;
    int error = -EINVAL;
    struct gpudma_mem_t *entry = NULL;

    printk(KERN_DEBUG"%s()\n", __FUNCTION__ );

    if(!drv || !param || !param->handle) {
        printk(KERN_DEBUG"%s(): Invalid parameters\n", __FUNCTION__ );
        goto do_out;
    }

    mutex_lock(&drv->m_mem_lock);

    entry = container_of(param->handle, struct gpudma_mem_t, mem_handle);
    if(entry && (entry->mem_id == MEM_ID)) {
        exist = true;
    }

    mutex_unlock(&drv->m_mem_lock);

    if(exist) {

        printk(KERN_DEBUG"%s(): %s - mem_owner_count: %d\n", __FUNCTION__, entry->mem_name, atomic_read(&entry->mem_owner_count) );

        //TODO: nvidia_p2p_put_pages(), nvidia_p2p_free_pages()
        atomic_dec(&entry->mem_lock_count);
        error = 0;

        printk(KERN_DEBUG"%s(): %s - unlocked %d\n", __FUNCTION__, entry->mem_name, atomic_read(&entry->mem_lock_count) );

    } else {

        printk(KERN_DEBUG"%s(): Invalid handle\n", __FUNCTION__ );
    }

do_out:
    return error;
}

//-----------------------------------------------------------------------------

int gpudma_mem_close( struct gpudma_driver *drv, struct gpudma_close_t *param )
{
    int error = -EINVAL;
    struct list_head *pos, *n;
    struct gpudma_mem_t *entry = NULL;
    struct gpudma_mem_t *handle = NULL;

    printk(KERN_DEBUG"%s()\n", __FUNCTION__ );

    if(!drv || !param || !param->handle) {
        printk(KERN_DEBUG"%s(): Invalid parameters\n", __FUNCTION__ );
        goto do_out;
    }

    mutex_lock(&drv->m_mem_lock);

    handle = container_of(param->handle, struct gpudma_mem_t, mem_handle);

    if(handle && (handle->mem_id == MEM_ID)) {

        list_for_each_safe(pos, n, &drv->m_mem_list) {

            entry = list_entry(pos, struct gpudma_mem_t, mem_list);

            if(entry == handle) {

                error = 0;

                if(atomic_dec_and_test(&entry->mem_owner_count)) {

                    printk(KERN_DEBUG"%s(): %s - deleted\n", __FUNCTION__, entry->mem_name );

                    list_del(pos);
                    kfree( (void*)entry );
                    break;

                } else {

                    printk(KERN_DEBUG"%s(): %s - object is using... skipping to delete it\n", __FUNCTION__, entry->mem_name );
                    error = -EBUSY;
                }
            }
        }
    }

    mutex_unlock(&drv->m_mem_lock);

do_out:
    return error;
}

//-----------------------------------------------------------------------------

int gpudma_mem_close_all( struct gpudma_driver *drv )
{
    int error = -EINVAL;
    int used_counter = 0;
    struct list_head *pos, *n;
    struct gpudma_mem_t *entry = NULL;

    printk(KERN_DEBUG"%s()\n", __FUNCTION__ );

    if(!drv) {
        printk(KERN_DEBUG"%s(): Invalid parameters\n", __FUNCTION__ );
        goto do_out;
    }

    mutex_lock(&drv->m_mem_lock);

    error = 0;

    list_for_each_safe(pos, n, &drv->m_mem_list) {

        entry = list_entry(pos, struct gpudma_mem_t, mem_list);

        if(atomic_read(&entry->mem_owner_count) == 0) {

            printk(KERN_DEBUG"%s(): %s - delete\n", __FUNCTION__, entry->mem_name );

        } else {

            printk(KERN_DEBUG"%s(): %s - using. forced deleting\n", __FUNCTION__, entry->mem_name );
            used_counter++;
        }

        list_del(pos);
        kfree( (void*)entry );
    }

    if(used_counter)
        error = -EBUSY;

    mutex_unlock(&drv->m_mem_lock);

do_out:
    return error;
}

//-----------------------------------------------------------------------------
