
#ifndef __GPUDMAPROC_H__
#define __GPUDMAPROC_H__

void gpudma_register_proc(char *name, void *fptr, void *data);
void gpudma_remove_proc(char *name);

#endif
