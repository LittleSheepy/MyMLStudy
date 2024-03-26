/* ��ѭ����������josephus������㷨*/

#include<stdio.h>
#include<stdlib.h>

#define  FALSE   0
#define  TRUE    1

typedef int DataType;
typedef struct Node *PNode;   /* ���ָ������ */

struct  Node {                           /* ��������ṹ */
    DataType info;
    PNode link; 
};

typedef struct Node *LinkList;

/*Ϊ���ݲ������㣬��������PlinkList���ͣ������������͵�ָ������*/
typedef  LinkList *PLinkList;

/* ��1��������nΪ*pclist��ʾ��ѭ�����ʼ�� */
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
        if (p == NULL) return FALSE; /* �˳�ǰӦ���ͷ������ѷ����� */
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
    
    /* �ҵ�s��Ԫ�أ����ú� pre �� p */
    p = list;
    if (s == 1)
        for (pre = p; pre->link != p; pre = pre->link)
            ;
    else for (i = 1; i < s; i++) {
        pre = p;   p = p->link; 
    }

    while (p != p->link) {              /* �������н���������1ʱ */
        for (i = 1; i < m; i++) {            /* �ҵ�m����� */
            pre = p;  
            p = p->link;
        }
        printf("out element: %d \n", p->info); /* ����ý�� */
        pre->link = p->link;             /* ɾ���ý�� */
        free(p);
        p = pre->link;     
    }

    printf("out element: %d \n", p->info);       /* ������һ����� */
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
    int n, s, m; /* ���������������ֵ */
    
    inputnsm(&n, &s, &m);
    josephus_clink(n, s, m);
    
    getchar(); getchar();
      
    return 0;
}


