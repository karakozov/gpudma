
KERNELVER := $(shell uname -r)
NVIDIA_DRIVER_PATH := $(HOME)/cuda/NVIDIA-Linux-x86_64
NVIDIA_DRIVER_VER := 367.44
NVIDIA_DRIVER := $(NVIDIA_DRIVER_PATH)-$(NVIDIA_DRIVER_VER)
KBUILD_EXTRA_SYMBOLS := $(NVIDIA_DRIVER)/kernel/Module.symvers
EXTRA_CFLAGS += -O2
EXTRA_CFLAGS += -I../../include
EXTRA_CFLAGS += -I$(HOME)/cuda/NVIDIA-Linux-x86_64-367.44/kernel/nvidia

ifneq ($(KERNELRELEASE),)

obj-m += gpudma.o
gpudma-objs := gpudmaproc.o ioctlrw.o gpudmadrv.o gpudmamem.o

else

KERNELDIR ?= /lib/modules/$(shell uname -r)/build

PWD := $(shell pwd)

all:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) EXTRA_CFLAGS="$(EXTRA_CFLAGS)" modules

endif

distclean:
	rm -rf *.o *~ core .depend .*.cmd *.ko *.mod.c .tmp_versions *.bak .*.cache *.d
clean:
	rm -rf *.o *~ core .depend .*.cmd *.ko *.mod.c .tmp_versions *.bak .*.cache *.d *.markers *.symvers *.order
