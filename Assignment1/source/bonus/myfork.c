#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <wait.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

int PidList[50];
int Signal[50];
int cnt = 0;

char SignalType[][10] = {"", "SIGHUP", "SIGINT", "SIGQUIT", "SIGILL", "SIGTRAP", "SIGABRT",
                          "SIGBUS", "SIGFPE", "SIGKILL", "", "SIGSEGV", "", "SIGPIPE",
						  "SIGALRM", "SIGTERM", "", "", "", "SIGSTOP"};
char SignalDescription[][30] = {"", "hang up", "interrupt", "quit", "illegal instruction",
                                "trap", "abort", "bus", "floating point exception", "kill",
                                "", "segment fault", "", "pipe", "alrm", "termination", "",
                                "", "", "stop"};

void Recur (int n, int rank, char* arg[])
{
    pid_t pid;
    int status;

    pid = vfork();

    if (pid == -1)
	{
        perror("fork");
        exit(1);
    }
    else
	{
        if (pid == 0)
		{
            if (rank < n)
                Recur(n, ++rank, arg);
            else
                execvp(arg[rank - 1], arg);
        }
        else
		{
            waitpid(pid, &status, WUNTRACED);

	    	if (cnt != 0)
	    		rank--;
	    	cnt++;

            Signal[rank] = status;
            PidList[rank] = pid;
            

            if (rank != 1)
			    execvp(arg[rank - 2], arg);
        }
    }
}


int main(int argc,char *argv[])
{
    int i;
	char *arg[argc];

    for(i = 0; i < argc - 1; i++)
        arg[i] = argv[i + 1];
    arg[argc - 1] = NULL;

    Recur(argc - 1, 1, arg);
    
    printf("The process tree: %d", getpid());
    for(i = 1; i < argc; i++)
        printf("->%d", PidList[i]);
    printf("\n");

    for(i = 1; i < argc; i++)
    {
        printf("The child process (pid=%d) ", PidList[argc - i]);
        
        if(i == argc - 1)
            printf("of parent process (pid=%d) ", getpid());
        else
            printf("of parent process (pid=%d) ", PidList[argc - 1 - i]);
        
        if (Signal[argc - i])
            printf("is terminated by signal\nIts signal number is %d\nchild process got %s signal\nchild was terminated by %s signal\n\n",
                   Signal[argc - i], SignalType[Signal[argc - i]], SignalDescription[Signal[argc - i]]);
        else
            printf("has normal execution\nIts exit status = 0\n\n");
    }

    printf("\nMyfork process(pid=%d) execute normally\n",getpid());

    return 0;
}