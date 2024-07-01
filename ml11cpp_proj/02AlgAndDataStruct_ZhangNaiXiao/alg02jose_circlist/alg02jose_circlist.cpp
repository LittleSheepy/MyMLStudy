/* 用循环单链表解决josephus问题的算法*/

#include<stdio.h>
#include<stdlib.h>

#define  FALSE   0
#define  TRUE    1

typedef int DataType;
typedef struct Node *PNode;   /* 结点指针类型 */

struct  Node {                           /* 单链表结点结构 */
    DataType info;
    PNode link; 
};

typedef struct Node *LinkList;

/*为传递参数方便，这里引入PlinkList类型，它是链表类型的指针类型*/
typedef  LinkList *PLinkList;

/* 用1，……，n为*pclist所示的循环表初始化 */
int initlist(PLinkList pclist, int n) {
    int i;
    PNode p, q;
    
    q = (PNode)malloc( sizeof( struct Node ) );
    if ( q == NULL ) return FALSE;
    *pclist = q;
    q->info = 1; 
    q->link = q;
    
    for(i = 2; i <= n; i++) {
        p = (PNode)malloc(sizeof(struct Node));
        if (p == NULL) return FALSE; /* 退出前应该释放所有已分配结点 */
        p->info = i;
        p->link = q->link;
        q->link = p;
        q = p;
    }
    return TRUE;
}

void josephus_clink( int n, int s, int m ) {
    LinkList list; 
    PNode p, pre;
    int i;
    
    if (initlist(&list, n) == FALSE) {
        printf("Out of space!\n");
        return;
    }
    
    /* 找第s个元素，设置好 pre 和 p */
    pre = p = list;
    if (s == 1)
        for (pre = p; pre->link != p; pre = pre->link)
            ;
    else for (i = 1; i < s; i++) {
        pre = p;   p = p->link; 
    }

    while (p != p->link) {              /* 当链表中结点个数大于1时 */
        for (i = 1; i < m; i++) {            /* 找第m个结点 */
            pre = p;  
            p = p->link;
        }
        printf("out element: %d \n", p->info); /* 输出该结点 */
        pre->link = p->link;             /* 删除该结点 */
        free(p);
        p = pre->link;     
    }

    printf("out element: %d \n", p->info);       /* 输出最后一个结点 */
    free(p);
}

void inputnsm(int* np, int* sp, int* mp){
    printf("please input the values of n = ");
    scanf_s("%d", np);
    printf("please input the values of s = ");
    scanf_s("%d", sp);
    printf("please input the values of m = ");     
    scanf_s("%d", mp);
}

int main( ){
    int n = 8, s = 1, m = 4; /* 输入所需各参数的值 */
    
    //inputnsm(&n, &s, &m);
    josephus_clink(n, s, m);
    
    getchar(); getchar();
      
    return 0;
}


