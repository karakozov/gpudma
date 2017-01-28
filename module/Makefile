
KERNELVER := $(shell uname -r)

ifndef GPUDMA_DIR

NVIDIA_DRIVER_PATH := $(HOME)/gpudma/nvidia

else

NVIDIA_DRIVER_PATH := $(GPUDMA_DIR)/nvidia

endif


KBUILD_EXTRA_SYMBOLS := $(NVIDIA_DRIVER_PATH)/kernel/Module.symvers
EXTRA_CFLAGS += -O2
EXTRA_CFLAGS += -I$(NVIDIA_DRIVER_PATH)/kernel/nvidia

ifneq ($(KERNELRELEASE),)

obj-m += gpumem.o
gpumem-objs := gpumemproc.o ioctlrw.o gpumemdrv.o

else

endif

KERNELDIR ?= /lib/modules/$(shell uname -r)/build


PWD := $(shell pwd)



all:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) EXTRA_CFLAGS="$(EXTRA_CFLAGS)" modules



distclean:
	rm -rf *.o *~ core .depend .*.cmd *.ko *.mod.c .tmp_versions *.bak .*.cache *.d
clean:
	rm -rf *.o *~ core .depend .*.cmd *.ko *.mod.c .tmp_versions *.bak .*.cache *.d *.markers *.symvers *.order
