
KERNELVER := $(shell uname -r)

ifndef GPUDMA_DIR

NVIDIA_DRIVER_PATH := $(HOME)/gpudma/nvidia

else

NVIDIA_DRIVER_PATH := $(GPUDMA_DIR)/nvidia

endif

KBUILD_EXTRA_SYMBOLS := $(NVIDIA_DRIVER_PATH)/kernel/Module.symvers
EXTRA_CFLAGS += -fno-stack-protector 
#EXTRA_CFLAGS += -fno-stack-protector -fno-stack-protector-strong
#EXTRA_CFLAGS += -fno-pie
EXTRA_CFLAGS += -O2
EXTRA_CFLAGS += -I$(NVIDIA_DRIVER_PATH)/kernel/nvidia

ccflags-y += $(EXTRA_CFLAGS)

ifneq ($(KERNELRELEASE),)

obj-m += gpumem.o
gpumem-objs := gpumemproc.o ioctlrw.o gpumemdrv.o

else

endif

KERNELDIR ?= /lib/modules/$(shell uname -r)/build


#PWD := $(shell pwd)
#PWD := $(PWD)



all:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules
#	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules



distclean:
	rm -rf *.o *~ core .depend .*.cmd *.ko *.mod.c .tmp_versions *.bak .*.cache *.d
clean:
	rm -rf *.o *~ core .depend .*.cmd *.ko *.mod.c .tmp_versions *.bak .*.cache *.d *.markers *.symvers *.order
