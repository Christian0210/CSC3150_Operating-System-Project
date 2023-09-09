#include <linux/module.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/kthread.h>
#include <linux/kernel.h>
#include <linux/err.h>
#include <linux/slab.h>
#include <linux/printk.h>
#include <linux/jiffies.h>
#include <linux/kmod.h>
#include <linux/fs.h>

MODULE_LICENSE("GPL");

static struct task_struct *task;

struct wait_opts 
{
	enum pid_type wo_type;
	int wo_flags;
	struct pid *wo_pid;
	struct siginfo __user *wo_info;
	int __user *wo_stat;
	struct rusage __user *wo_rusage;
	wait_queue_t child_wait;
	int notask_error ;
};

extern long do_wait(struct wait_opts *wo);
extern long _do_fork(unsigned long clone_flags,
		             unsigned long stack_start,
		             unsigned long stack_size,
		             int __user *parent_tidptr,
		             int __user *child_tidptr,
		             unsigned long tls);
extern int do_execve(struct filename *filename,
		             const char __user *const __user *__argv,
		             const char __user *const __user *__envp);
extern struct filename *getname(const char __user * filename);


int my_exec(void)
{
	int result;
	const char path[] = "/home/seed/Desktop/source/program2/test";
	const char *const argv[] = {path,NULL,NULL};
	const char *const envp[] = {"HOME=/","PATH=/sbin:/user/sbin:/bin:/usr/bin",NULL};
	
    struct filename *my_filename = getname(path);

    printk("[Program2] : child process\n");

	result = do_execve(my_filename, argv, envp);
	if(!result)
        return 0;
	do_exit(result);
}

void my_wait(pid_t pid)
{
	char Signal[][10] = {"", "SIGHUP", "SIGINT", "SIGQUIT", "SIGILL", "SIGTRAP", "SIGABRT",
                         "SIGBUS", "SIGFPE", "SIGKILL",	"SIGUSR1", "SIGSEGV", "SIGUSR2",
                         "SIGPIPE", "SIGALRM", "SIGTERM", "SIGSTKFLT", "SIGCHLD", "SIGCONT",
                         "SIGSTOP",	"SIGTSTP", "SIGTTIN", "SIGTTOU", "SIGURG", "SIGXCPU",
                         "SIGXFSZ",	"SIGVTALRM", "SIGPROF", "SIGWINCH", "SIGIO", "SIGPWR",
                         "SIGSYS", "SIGRTMIN"};
	int status;
	struct wait_opts wo;
	struct pid *wo_pid=NULL;
	enum pid_type type;

	type = PIDTYPE_PID;
	wo_pid = find_get_pid(pid);
	wo.wo_type = type;
	wo.wo_pid = wo_pid;
	wo.wo_flags = WEXITED;
	wo.wo_info = NULL;
	wo.wo_stat = (int __user*)&status;
	wo.wo_rusage = NULL;
	do_wait(&wo);

    if (*wo.wo_stat == 17){
        printk("[program2] : Normal termination with EXIT STATUS = %d\n", *wo.wo_stat);
    }
    else if (*wo.wo_stat>0)
    {
        printk("[program2] : get %s signal", Signal[*wo.wo_stat]);
        if (*wo.wo_stat == 19)
            printk("[program2] : child process stopped\n");
        else 
            printk("[program2] : child process has %s error", Signal[*wo.wo_stat]);
	}

	printk("[program2] : The return signal is %d\n",*wo.wo_stat);
	return;
}

int my_fork(void *argc)
{
	int i;
	pid_t pid;
	struct k_sigaction *k_action = &current->sighand->action[0];

	for(i = 0; i < _NSIG; i++)
    {
		k_action->sa.sa_handler = SIG_DFL;
		k_action->sa.sa_flags = 0;
		k_action->sa.sa_restorer = NULL;
		sigemptyset(&k_action->sa.sa_mask);
		k_action++;
	}

	pid = _do_fork(SIGCHLD,(unsigned long)&my_exec, 0, NULL, NULL, 0);	
	printk("[program2] : The child process has pid = %d\n", pid);
	printk("[program2] : This is the parent process, pid = %d\n", (int)current->pid);
	my_wait((int)pid);

	return 0;
}

static int __init program2_init(void)
{
	printk("[program2] : module_init\n");
	task = kthread_create(&my_fork,NULL, "MyThread");
	printk("[program2] : module_init create kthread start");
	
    if(!IS_ERR(task))
    {
		printk("[program2] : module_init kernel thread start\n");
		wake_up_process(task);
	}

	return 0;
}

static void __exit program2_exit(void)
{
	printk("[program2] : module_exit\n");
}

module_init(program2_init);
module_exit(program2_exit);