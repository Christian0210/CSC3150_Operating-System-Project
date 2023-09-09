#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/stat.h>
#include <linux/fs.h>
#include <linux/workqueue.h>
#include <linux/sched.h>
#include <linux/interrupt.h>
#include <linux/slab.h>
#include <linux/cdev.h>
#include <linux/delay.h>
#include <asm/uaccess.h>
#include "ioc_hw5.h"

MODULE_LICENSE("GPL");

#define PREFIX_TITLE "OS_AS5"
#define DEV_NAME "mydev"
#define DEV_BASEMINOR 0
#define DEV_COUNT 1
static int dev_major;
static int dev_minor;
static int interrupt_count = 0;
static int IRQ_NUM = 1;
void *irq_dev_id = (void *)&IRQ_NUM;
struct cdev *dev_cdev;

// DMA
#define DMA_BUFSIZE 64
#define DMASTUIDADDR 0x0        // Student ID
#define DMARWOKADDR 0x4         // RW function complete
#define DMAIOCOKADDR 0x8        // ioctl function complete
#define DMAIRQOKADDR 0xc        // ISR function complete
#define DMACOUNTADDR 0x10       // interrupt count function complete
#define DMAANSADDR 0x14         // Computation answer
#define DMAREADABLEADDR 0x18    // READABLE variable for synchronize
#define DMABLOCKADDR 0x1c       // Blocking or non-blocking IO
#define DMAOPCODEADDR 0x20      // data.a opcode
#define DMAOPERANDBADDR 0x21    // data.b operand1
#define DMAOPERANDCADDR 0x25    // data.c operand2
void *dma_buf;

static ssize_t drv_read(struct file *filp, char __user *buffer, size_t, loff_t*);
static int drv_open(struct inode*, struct file*);
static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t, loff_t*);
static int drv_release(struct inode*, struct file*);
static long drv_ioctl(struct file *, unsigned int , unsigned long );

static struct file_operations fops = {
      owner: THIS_MODULE,
      read: drv_read,
      write: drv_write,
      unlocked_ioctl: drv_ioctl,
      open: drv_open,
      release: drv_release,
};

static struct work_struct *work;

struct DataIn
{
    char a;
    int b;
    short c;
} *dataIn;


static void drv_arithmetic_routine(struct work_struct* ws);


void myoutc(unsigned char data, unsigned short int port){
    *(volatile unsigned char*)(dma_buf + port) = data;
}


void myouts(unsigned short data, unsigned short int port) {
    *(volatile unsigned short*)(dma_buf + port) = data;
}


void myouti(unsigned int data, unsigned short int port) {
    *(volatile unsigned int*)(dma_buf + port) = data;
}


unsigned char myinc(unsigned short int port) {
    return *(volatile unsigned char*)(dma_buf + port);
}


unsigned short myins(unsigned short int port) {
    return *(volatile unsigned short*)(dma_buf + port);
}


unsigned int myini(unsigned short int port) {
    return *(volatile unsigned int*)(dma_buf+port);
}


static int drv_open(struct inode* ii, struct file* ff)
{
	try_module_get(THIS_MODULE);
    printk("%s:%s(): device open\n", PREFIX_TITLE, __func__);
	return 0;
}


static int drv_release(struct inode* ii, struct file* ff)
{
	module_put(THIS_MODULE);
    printk("%s:%s(): device close\n", PREFIX_TITLE, __func__);
	return 0;
}


static ssize_t drv_read(struct file *filp, char __user *buffer, size_t ss, loff_t* lo) 
{
    if (myini(DMAREADABLEADDR) == 1)
    {
        printk("%s:%s(): ans = %i\n", PREFIX_TITLE, __func__, myini(DMAANSADDR));
        put_user(myini(DMAANSADDR), (int*)buffer);
        myouti(0, DMAREADABLEADDR);
        myouti(0, DMASTUIDADDR);
        myouti(0, DMARWOKADDR);
        myouti(0, DMAIOCOKADDR);
        myouti(0, DMAIRQOKADDR);
        myouti(0, DMACOUNTADDR);
        myouti(0, DMAANSADDR);
        myouti(0, DMABLOCKADDR);
        myouti(0, DMAOPCODEADDR);
        myoutc(' ', DMAOPCODEADDR);
        myouti(0, DMAOPERANDCADDR);
        myouti(0, DMAOPERANDBADDR);
    }
	return 0;
}


static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t ss, loff_t* lo) 
{
    struct DataIn data;
    get_user(data.a, (char*)buffer);
    get_user(data.b, (int*)buffer + 1);
    get_user(data.c, (int*)buffer + 2);
    myoutc(data.a, DMAOPCODEADDR);
    myouti(data.b,DMAOPERANDBADDR);
    myouti(data.c,DMAOPERANDCADDR);

    INIT_WORK(work, drv_arithmetic_routine);

    printk("%s:%s(): queue work\n", PREFIX_TITLE, __func__);

    if (myini(DMABLOCKADDR))
    {
        printk("%s:%s(): block\n", PREFIX_TITLE, __func__);
        schedule_work(work);
        flush_scheduled_work();
    }
    else 
    {
        printk("%s:%s(): non-blocking\n", PREFIX_TITLE, __func__);
        myouti(0, DMAREADABLEADDR);
        schedule_work(work);
    }
	return 0;
}


static long drv_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    int info, reads, ret;
    ret = get_user(info, (int*)arg);
    
    if (ret < 0)
        return ret;

    switch(cmd) 
    {
        case HW5_IOCSETSTUID:
            myouti(info, DMASTUIDADDR);
            printk("%s:%s(): My STUID is = %i\n", PREFIX_TITLE, __func__, info);
            break;
        case HW5_IOCSETRWOK:
            if (info == 0 || info == 1)
            {
                printk("%s:%s(): RW OK\n", PREFIX_TITLE, __func__);
                myouti(info, DMARWOKADDR);
            }
            else
            {
                printk("%s:%s(): RW failed\n", PREFIX_TITLE, __func__);
                return -1;
            }
            break;
        case HW5_IOCSETIOCOK:
            if (info == 0 || info == 1)
            {
                myouti(info, DMAIOCOKADDR);
                printk("%s:%s(): IOC OK\n", PREFIX_TITLE, __func__);
            }
            else
            {
                printk("%s:%s(): IOC failed\n", PREFIX_TITLE, __func__);
                return -1;
            }
            break;
        case HW5_IOCSETIRQOK:
            if (info == 0 || info == 1)
            {
                myouti(info, DMAIRQOKADDR);
                printk("%s:%s(): IRQ OK\n", PREFIX_TITLE, __func__);
            }
            else
            {
                printk("%s:%s(): IRQ failed\n", PREFIX_TITLE, __func__);
                return -1;
            }
            break;
        case HW5_IOCSETBLOCK:
            if (info == 0 || info == 1) 
                myouti(info, DMABLOCKADDR);
            else
                return -1;
            if (info) 
                printk("%s:%s(): Blocking IO\n", PREFIX_TITLE, __func__);
            else 
                printk("%s:%s(): Non-Blocking IO\n", PREFIX_TITLE, __func__);
            break;
        case HW5_IOCWAITREADABLE:
            reads = myini(DMAREADABLEADDR);
            while (!reads)
            {
                msleep(6000);
                reads = myini(DMAREADABLEADDR);
            }
            put_user(reads, (int *)arg);
            printk("%s:%s(): wait readable 1\n", PREFIX_TITLE, __func__);
            break;
        default:
            printk("%s:%s(): unknown error\n", PREFIX_TITLE, __func__);
    }
    return 0;
}


int prime(int base, short nth)
{
    int cnt=0;
    int i, n, tag;
    n = base;

    while(cnt < nth)
    {
        tag = 1;
        n++;
        if (n % 2)
        {
            for(i = 3; i * i <= n; i += 2)
                if(n % i == 0)
                {
                    tag = 0;
                    break;
                }
        }
        else
            tag = 0;
        cnt += tag;
    }
    return n;
}


static void drv_arithmetic_routine(struct work_struct* ws)
{
    struct DataIn data;
    int ans;

    data.a = myinc(DMAOPCODEADDR);
    data.b = myini(DMAOPERANDBADDR);
    data.c = myini(DMAOPERANDCADDR);

    switch(data.a) 
    {
        case '*':
            ans=data.b*data.c;
            break;
        case '/':
            ans=data.b/data.c;
            break;
        case '+':
            ans=data.b+data.c;
            break;
        case '-':
            ans=data.b-data.c;
            break;
        case 'p':
            ans = prime(data.b, data.c);
            break;
        default:
            ans=0;
    }

    myouti(ans,DMAANSADDR);
    myouti(1, DMAREADABLEADDR);
    printk("%s:%s(): %i %c %i = %i\n", PREFIX_TITLE,__func__,data.b, data.a, data.c, ans);
}


static irqreturn_t irqcnt(int irq, void* dev_id)
{
    if (irq == IRQ_NUM)
        interrupt_count++;
    return IRQ_NONE;
}


static int __init init_modules(void)
{
    dev_t dev;

    printk("%s:%s():...............Start...............\n", PREFIX_TITLE, __func__);

    printk("%s:%s(): request_irq 1 return %i\n", PREFIX_TITLE, __func__,
        request_irq(1, irqcnt, IRQF_SHARED, "myinterrupts", irq_dev_id));

    if (alloc_chrdev_region(&dev, DEV_BASEMINOR, DEV_COUNT, DEV_NAME) < 0)
    {
        printk(KERN_ALERT"Register chrdev failed!\n");
        return -1;
    }
    else
        printk("%s:%s(): register chrdev(%i,%i)\n", PREFIX_TITLE, __func__, MAJOR(dev), MINOR(dev));
    
    dev_major = MAJOR(dev);
    dev_minor = MINOR(dev);

    dev_cdev = cdev_alloc();
    cdev_init(dev_cdev, &fops);
    dev_cdev->ops = &fops;
    dev_cdev->owner = THIS_MODULE;

    if(cdev_add(dev_cdev,dev,1) < 0)
    {
        printk(KERN_ALERT"%s:%s(): Add cdev failed!\n",PREFIX_TITLE,__func__);
        return -1;
    }

    printk("%s:%s(): allocate dma buffer\n",PREFIX_TITLE,__func__);
    dma_buf = kmalloc(DMA_BUFSIZE, GFP_KERNEL);

    work = kmalloc(sizeof(typeof(*work)),GFP_KERNEL);
	return 0;
}


static void __exit exit_modules(void)
{
    free_irq(1,irq_dev_id);
    printk("%s:%s(): interrupt count = %d\n", PREFIX_TITLE, __func__, interrupt_count);
    kfree(dma_buf);
    printk("%s:%s(): free dma buffer\n", PREFIX_TITLE, __func__);

    unregister_chrdev_region(MKDEV(dev_major,dev_minor), DEV_COUNT);
    cdev_del(dev_cdev);
    printk("%s:%s(): unregister chrdev\n", PREFIX_TITLE, __func__);
    kfree(work);
    printk("%s:%s():..............End..............\n", PREFIX_TITLE, __func__);
}


module_init(init_modules);
module_exit(exit_modules);
