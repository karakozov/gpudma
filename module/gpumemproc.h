
#ifndef __GPUDMAPROC_H__
#define __GPUDMAPROC_H__

void gpumem_register_proc(char *name, void *fptr, void *data);
void gpumem_remove_proc(char *name);

#endif
