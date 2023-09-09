#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/wait.h>
#include <sys/types.h>

int main (int argc, char *argv[])
{    
    int status, sig_number;
	pid_t pid;
	char SignalArray[][10] = {"", "SIGHUP", "SIGINT", "SIGQUIT", "SIGILL", "SIGTRAP",
							  "SIGABRT", "SIGBUS", "SIGFPE", "SIGKILL", "", "SIGSEGV",
							  "", "SIGPIPE", "SIGALRM", "SIGTERM", "", "", "", "SIGSTOP"};
	char SignalDescription[][30] = {"", "hangup", "interrupt", "quit signal",
									"illegal instruction", "trace trap", "abort", "bus error",
									"floating-point exception", "kill, unblockable", "",
									"segmentation violation", "", "broken pipe", "alarm clock",
									"termination"};
	
    printf ("Process start to fork\n");

    pid = fork();
    
    if(pid < 0)
    {   
		perror ("fork error!\n");
		exit(1);
	}
    else 
	{
        if(pid == 0)
	    {     
            int i;
            char *arg[argc];

            printf("I'm the parent process, my pid = %d\nI'm the child process, my pid = %d\n",
				   getppid(), getpid());

            for(i = 0; i < argc - 1; i++)
                arg[i] = argv[i + 1];
            arg[argc - 1] = NULL;
            
            printf("Child process start to execute test program:\n");
            raise(SIGCHLD);

            execvp(arg[0], arg);
            perror("execve");
            exit(EXIT_FAILURE);
        }
        else
		{
            waitpid(-1, &status, WUNTRACED);
            printf("Parent process receving the SIGCHLD signal\n");

            if(WIFEXITED(status))
            	printf("Normal termination with EXIT STATUS = %d\n", WEXITSTATUS(status));
            else if(WIFSIGNALED(status))
			{
            	sig_number = WTERMSIG(status);

				if(sig_number > 0 && sig_number < 16 && sig_number != 10 && sig_number != 12)
					printf("child process get %s signal\nchild process is abort by %s signal\n",
					       SignalArray[sig_number], SignalDescription[sig_number]);
				else
		        	printf("Receiving an unknown signal!\nSignal Number: %d\n", WTERMSIG(status));

            	printf("CHILD EXECUTION FAILED!!\n");
            }
            else if(WIFSTOPPED(status))
            	printf("child process get %s signal\nchild process stopped\nCHILD PROCESS STOPPED\n",
				       SignalArray[WSTOPSIG(status)]);
            else
            	printf("CHILD PROCESS CONTINUED\n");

            exit(1);
        }
    }

    return 0;
}
