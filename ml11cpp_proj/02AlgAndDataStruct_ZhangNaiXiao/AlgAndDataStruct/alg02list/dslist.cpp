/* 线性表的顺序表示：函数实现*/

#include <stdio.h>
#include <stdlib.h>

#include "dslist.h"

PDySeqList createNullList_seq( void ) {
    PDySeqList lp = (PDySeqList)malloc(sizeof(struct DySeqList));
    if (lp != NULL) {
        lp->elems = (DataType*)malloc(sizeof(DataType)*NBASE);
        if (lp->elems) {
            lp->n = 0;          /*空表长度为0 */
            lp->nmax = NBASE;
            return lp;
        }
        else free(lp);
    }
    printf("Out of space!\n");  /*存储分配失败*/
    return NULL;
}

/*在palist所指顺序表中下标为p的元素之前插入元素x*/
int insert_seq(PDySeqList lp, int p, DataType x) {
    int q;
    if ( p < 0 || p > lp->n ) { /* 不存在下标为p的元素 */
        printf("Index of seq-list is out of range! \n");
        return FALSE;
    }
    if ( lp->n == lp->nmax ) { 	/* 存储区满 */
        DataType *dp =          /* 存储区扩大一倍 */
            (DataType*)realloc(lp->elems, lp->nmax*2*sizeof(DataType));
        if (dp == NULL) { 		/* 空间耗尽，原来元素还在原处 */
            printf("Seq-list overflow!\n");
            return FALSE;
        }
        lp->elems = dp;
        lp->nmax *= 2;
    }
    
    for(q = lp->n - 1; q >= p; q--)  	/* 插入位置及之后的元素均后移一个位置 */
        lp->elems[q+1] = lp->elems[q];
        
    lp->elems[p] = x;				/* 插入元素x */
    lp->n++;			/* 元素个数加1 */
    return TRUE;
}

/*在lp所指顺序表中删除下标为ｐ的元素*/
int delete_seq(PDySeqList lp, int p ) {
    int q;
    if (p < 0 || p > lp->n-1 ) { 	/* 不存在下标为p的元素 */
        printf("Index of seq-list is out of range!\n ");
        return FALSE;
    }
	
    for(q = p; q < lp->n - 1; q++) 	/* 被删除元素之后的元素均前移一个位置 */
	    lp->elems[q] = lp->elems[q+1];
	    
    lp->n--;			/* 元素个数减1 */
    return TRUE;
}

/*求x在lp所指顺序表中的下标*/
int locate_seq(PDySeqList lp, DataType x) {
    int q;
    for ( q = 0; q < lp->n; q++ )
        if (lp->elems[q] == x) return q;
    return  -1;
}

/* 求lp所指顺序表中下标为p的元素值 */
DataType  retrieve_seq(PDySeqList lp, int p ) {
    if ( p >= 0 && p < lp->n )	/* 存在下标为p的元素 */
        return lp->elems[p];
    	
    printf("Index of seq-list is out of range.\n ");
    return SPECIAL;                 /* 返回一个顺序表中没有的特殊值 */
}

int isNullList_seq(PDySeqList lp ) {
    return lp->n == 0;
}


