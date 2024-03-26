/* 用顺序表解决josephus问题的算法*/

#include<stdio.h>
#include<stdlib.h>

#define  MAXNUM  100
#define  FALSE   0
#define  TRUE    1

typedef  int  DataType;

struct SeqList {
    int       n;					/* 存放线性表中元素的个数 n < MAXNUM  */
  	DataType  element[MAXNUM];	/* 存放线性表中的元素 */
};

typedef  struct SeqList *PSeqList;

PSeqList createNullList_seq( void ) {
    PSeqList palist = (PSeqList)malloc(sizeof(struct SeqList));
    if (palist != NULL)
        palist ->n = 0;					/* 空表长度为0 */
    else
        printf("Out of space!!\n");    	/* 存储分配失败 */
    return palist;
}

/* 在palist所指顺序表中下标为p的元素之前插入元素x */
int insert_seq( PSeqList palist, int p, DataType x) {
    int q;
    if ( palist->n == MAXNUM ) {		/* 溢出 */
        printf("Seq-list overflow!\n");
        return FALSE;
    }
    if (  p < 0  ||  p > palist->n  ) { 	/* 不存在下标p */
        printf("Index out of range! \n");
        return FALSE;
    }
    
    for (q = palist->n - 1; q >= p; q--) /* 插入位置及之后的元素均后移一个位置 */
        palist->element[q+1] = palist->element[q];
    palist->element[p] = x;	        /* 插入元素x */
    palist->n++;			        /* 元素个数加1 */
    return TRUE;
}

/* 在palist所指顺序表中删除下标为ｐ的元素 */
int  delete_seq( PSeqList palist, int p ) {
    int q;
    if (  p < 0  ||  p > palist->n  ) { 	/* 不存在下标p */
        printf("Index out of range! \n");
        return FALSE;
    }
    
    for (q = p; q < palist->n-1; q++) /* 被删除元素之后的元素均前移一个位置 */
        palist->element[q] = palist->element[q+1];
    palist->n--;				/* 元素个数减1 */
    return TRUE;
}

/* 求palist所指顺序表中下标为p的元素值 */
DataType  retrieve_seq( PSeqList palist, int p ) {
    if ( p >= 0 && p < palist->n  )	/* 存在下标为p的元素 */
        return palist->element[p];
    printf("Index out of range! \n ");
    return -1;   /* 返回一个顺序表中没有的特殊值 */
}

int init_jlist(PSeqList slist, int n) {
    int i,k;

    if (slist == NULL) return FALSE;
    if (n < 1 || n > MAXNUM) {
        printf("Number of elements is out of range!\n");
        return FALSE;
    }
    for ( i = 0; i < n; i++ ) 
        k = insert_seq( slist, i, i+1);
}

void josephus_seq( int n, int s, int m) {
    int s1, i, w;
    PSeqList jlist;
    s1 = s - 1;

    jlist = createNullList_seq();    /* 创建空顺序表 */
    if (jlist == NULL) return;
    if (init_jlist(jlist, n) == FALSE) return;
    /* 找出列的元素 */
    for (i = jlist->n; i > 0; i--) {
        s1 = ( s1 + m - 1 ) % i ;
        w = retrieve_seq(jlist, s1);     /* 求下标为s1的元素的值 */
        printf("Out element %d \n", w);   /* 元素出列 */
        delete_seq(jlist, s1);           /* 删除出列的元素 */
    }
    
    free(jlist);
}

/* ====================== */ 

void inputnsm(int* np, int* sp, int* mp) {
    printf("\n please input the values(<100) of n = ");
    scanf_s("%d", np);
    printf("\n please input the values of s = ");
    scanf_s("%d", sp);
    printf("\n please input the values of m = ");
    scanf_s("%d", mp);
}

int main( ){
    int n,s,m;

    inputnsm(&n, &s, &m);
    josephus_seq(n, s, m);
    
    getchar(); getchar();
    
    return 0;
}

