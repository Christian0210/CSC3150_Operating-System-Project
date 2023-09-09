#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>
#include <iostream>
using namespace std;

#define CONSOLE_CLEAR "\033[H\033[2J"
#define ROW 10
#define COLUMN 50

const int WIN = 1, LOSE = -1, QUIT = -2, NORMAL = 0;
int speed = 110000, inter = 25000;
int minspeed = speed - 2 * inter, maxspeed =  speed + 2 * inter;
bool game_continue = true, game_quit = false, game_pause = false;
int cursor[2];
char map[ROW+10][COLUMN], speed_type[][10] = {"Fastest", "Fast", "Normal", "Slow", "Slowest"};

pthread_mutex_t frog_mutex;
pthread_cond_t frog_threshold_cv;
pthread_t threads[2];

struct Node
{
    int x, y;
    Node(int _x, int _y ) : x( _x ), y( _y ) {};
    Node() {};
}frog;

int kbhit(void)
{
    struct termios oldt, newt;
    int ch;
    int oldf;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);
    ch = getchar();
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);
    if(ch != EOF)
    {
        ungetc(ch, stdin);
        return 1;
    }
    return 0;
}

void init()
{
    int i, j, s, l;
    
    for(i = 1; i <= ROW; i++)
        for(j = 0; j < COLUMN - 1; j++)
            map[i][j] = ' ';
    
    for(j = 0; j < COLUMN - 1; j++)
        map[ROW][j] = map[0][j] = '|' ;
    
    frog = Node(ROW, (COLUMN - 1) / 2) ;

    srand((unsigned)time(NULL));
    for(i = 1; i < ROW; i++)
    {
        s = rand() % (COLUMN - 1);
        l = rand() % ((COLUMN - 1) / 4) + (COLUMN - 1) / 4;
        for (j = s; j < s + l; j++)
            map[i][j % (COLUMN - 1)] = '=';
    }
}

int check(int x, int y)
{
    if (game_quit)
        return QUIT;
    if(y >= COLUMN - 1 || y < 0)
        return LOSE;
    if(x == 0) 
        return WIN;
    if(x == ROW || map[x][(y + cursor[x % 2]) % (COLUMN - 1)] == '=')
        return NORMAL;
    else
        return LOSE;
}

void* frog_move(void *idp)
{
    int *my_id = (int*)idp;
    char dir;

    while (game_continue)
    {
        pthread_mutex_lock(&frog_mutex);

        if (kbhit())
        {
            dir = getchar();
            if(dir == 'p' || dir == 'P')
                game_pause = !game_pause;
            if (!game_pause)
            {
                if((dir == 'w' || dir == 'W') && frog.x != 0)
                    frog.x--;
                if((dir == 's' || dir == 'S') && frog.x != ROW)
                    frog.x++;
                if(dir == 'a' || dir == 'A')
                    frog.y--;
                if(dir == 'd' || dir == 'D') 
                    frog.y++;
            }
            if((dir == 'k' || dir == 'K') && (speed > minspeed))
                speed -= inter;
            if((dir == 'j' || dir == 'J') && (speed < maxspeed))
                speed += inter;
            if(dir == 'q' || dir == 'Q')
                game_quit = true;
            if (check(frog.x, frog.y) != NORMAL)
                game_continue = false; 
        }

        pthread_cond_signal(&frog_threshold_cv);
        pthread_mutex_unlock(&frog_mutex);
    }
    pthread_cancel(threads[1]);
    pthread_cond_signal(&frog_threshold_cv);
    pthread_exit(NULL);
}

void print_map()
{
    int i, j;
    printf("[W], [A], [S], [D] :  move up, left, down, right.\n");
    printf("[J]                :  slow down the log.\n");
    printf("[K]                :  speed up the log.\n");
    printf("[P]                :  pause or continue the game.\n");
    printf("[Q]                :  quit the game.\n\n");
    for(i = 0; i <= ROW; i++)
        {   
            for(j = 0; j < COLUMN - 1; j++)
                if (i == frog.x && j == frog.y)
                    printf("0");
                else
                    printf("%c", map[i][(j + cursor[i % 2]) % (COLUMN - 1)]); 
            printf("\n");
        }
    printf("\nLog Moving Speed:  |");
    for(i = 0; i < (maxspeed - speed) / inter; i++)
        printf("-");
    printf("+");
    for(i = 0; i < (speed - minspeed) / inter; i++)
        printf("-");
    printf("|  %s\n", speed_type[(speed - minspeed) / inter]);
}

void* logs_move(void *idp)
{    
    int *my_id = (int*)idp;
    int i, j;

    while (game_continue)
    {
        pthread_mutex_lock(&frog_mutex);
        pthread_cond_wait(&frog_threshold_cv, &frog_mutex);
        usleep(speed);

        if(!game_pause)
        {
            cursor[1] = (cursor[1] - 1 + COLUMN - 1) % (COLUMN - 1);
            cursor[0] = (cursor[0] + 1) % (COLUMN - 1);
            if(frog.x > 0 && frog.x < ROW)    
                if (frog.x % 2)
                    frog.y++;
                else
                    frog.y--;
            if(check(frog.x, frog.y) != NORMAL)
                game_continue = false;
        }

        printf(CONSOLE_CLEAR);
        if(game_continue)
            print_map();

        pthread_mutex_unlock(&frog_mutex);
    }
    pthread_cancel(threads[0]);
    pthread_exit(NULL);
}

int main(int argc, char *argv[])
{    
    init();

    pthread_attr_t attr;
    pthread_mutex_init(&frog_mutex,NULL);
    pthread_cond_init(&frog_threshold_cv,NULL);
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);
    pthread_create(&threads[0],&attr,logs_move, (void*)0);
    pthread_create(&threads[1],&attr,frog_move, (void*)1);
    pthread_join(threads[0],NULL);
    pthread_join(threads[1],NULL);
    
    int status = check(frog.x,frog.y);
    if (status == WIN)
        printf("You win the game!!\n");
    if (status == LOSE)
        printf("You lose the game!!\n");
    if (status == QUIT)
        printf("You exit the game!!\n");
    
    pthread_attr_destroy(&attr);
    pthread_mutex_destroy(&frog_mutex);
    pthread_cond_destroy(&frog_threshold_cv);
    pthread_exit(NULL);
    return 0;
}
