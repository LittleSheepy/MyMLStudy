/* 线性表的顺序表示：函数实现*/

#include <stdio.h>
#include <stdlib.h>

#include "alg02list/slist.h"

PSeqList createNullList_seq(int m) {
    PSeqList palist = (PSeqList)malloc(sizeof(struct SeqList));
    if (palist != NULL){
        palist->element = (DataType*)malloc(sizeof(DataType) * m);
        if (palist->element) {
            palist->MAXNUM = m;
            palist->n = 0;      /*空表长度为0 */
            return palist;
        }
        else {
            free(palist);
        }
    }
    printf("Out of space!\n");    	/*存储分配失败*/
    return NULL;
}

/*在palist所指顺序表中下标为p的元素之前插入元素x*/
int insert_seq(PSeqList palist, int p, DataType x) {
    int q;
    if ( palist->n == palist->MAXNUM ) { 			/* 溢出 */
        printf("Seq-list overflow!\n");
        return FALSE;
    }
    if (  p < 0  ||  p > palist->n  ) { 	/* 不存在下标为p的元素 */
        printf("Index of seq-list is out of range! \n");
        return FALSE;
    }

    for(q = palist->n - 1; q >= p; q--)  	/* 插入位置及之后的元素均后移一个位置 */
        palist->element[q+1] = palist->element[q];

    palist->element[p] = x;				/* 插入元素x */
    palist->n++;			/* 元素个数加1 */
    return TRUE;
}

/*在palist所指顺序表中删除下标为ｐ的元素*/
int delete_seq( PSeqList palist, int p ) {
    int q;
    if (p < 0 || p > palist->n-1 ) { 	/* 不存在下标为p的元素 */
        printf("Index of seq-list is out of range!\n ");
        return FALSE;
    }

    for(q = p; q < palist->n-1; q++) 	/* 被删除元素之后的元素均前移一个位置 */
	    palist->element[q] = palist->element[q+1];

    palist->n--;			/* 元素个数减1 */
    return TRUE;
}

/*求x在palist所指顺序表中的下标*/
int locate_seq(PSeqList palist, DataType x) {
    int q;
    for ( q = 0; q<palist->n; q++ )
        if (palist->element[q] == x) return q;
    return  -1;
}

/* 求palist所指顺序表中下标为p的元素值 */
DataType  retrieve_seq( PSeqList palist, int p ) {
    if ( p >= 0 && p < palist->n )	/* 存在下标为p的元素 */
        return palist->element[p];

    printf("Index of seq-list is out of range.\n ");
    return SPECIAL;                 /* 返回一个顺序表中没有的特殊值 */
}

int isNullList_seq( PSeqList palist ) {
    return palist->n == 0;
}


