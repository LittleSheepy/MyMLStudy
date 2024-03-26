/* ��˳�����josephus������㷨*/

#include<stdio.h>
#include<stdlib.h>

#define  MAXNUM  100
#define  FALSE   0
#define  TRUE    1

typedef  int  DataType;

struct SeqList {
    int       n;					/* ������Ա���Ԫ�صĸ��� n < MAXNUM  */
  	DataType  element[MAXNUM];	/* ������Ա��е�Ԫ�� */
};

typedef  struct SeqList *PSeqList;

PSeqList createNullList_seq( void ) {
    PSeqList palist = (PSeqList)malloc(sizeof(struct SeqList));
    if (palist != NULL)
        palist ->n = 0;					/* �ձ���Ϊ0 */
    else
        printf("Out of space!!\n");    	/* �洢����ʧ�� */
    return palist;
}

/* ��palist��ָ˳������±�Ϊp��Ԫ��֮ǰ����Ԫ��x */
int insert_seq( PSeqList palist, int p, DataType x) {
    int q;
    if ( palist->n == MAXNUM ) {		/* ��� */
        printf("Seq-list overflow!\n");
        return FALSE;
    }
    if (  p < 0  ||  p > palist->n  ) { 	/* �������±�p */
        printf("Index out of range! \n");
        return FALSE;
    }
    
    for (q = palist->n - 1; q >= p; q--) /* ����λ�ü�֮���Ԫ�ؾ�����һ��λ�� */
        palist->element[q+1] = palist->element[q];
    palist->element[p] = x;	        /* ����Ԫ��x */
    palist->n++;			        /* Ԫ�ظ�����1 */
    return TRUE;
}

/* ��palist��ָ˳�����ɾ���±�Ϊ���Ԫ�� */
int  delete_seq( PSeqList palist, int p ) {
    int q;
    if (  p < 0  ||  p > palist->n  ) { 	/* �������±�p */
        printf("Index out of range! \n");
        return FALSE;
    }
    
    for (q = p; q < palist->n-1; q++) /* ��ɾ��Ԫ��֮���Ԫ�ؾ�ǰ��һ��λ�� */
        palist->element[q] = palist->element[q+1];
    palist->n--;				/* Ԫ�ظ�����1 */
    return TRUE;
}

/* ��palist��ָ˳������±�Ϊp��Ԫ��ֵ */
DataType  retrieve_seq( PSeqList palist, int p ) {
    if ( p >= 0 && p < palist->n  )	/* �����±�Ϊp��Ԫ�� */
        return palist->element[p];
    printf("Index out of range! \n ");
    return -1;   /* ����һ��˳�����û�е�����ֵ */
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

    jlist = createNullList_seq();    /* ������˳��� */
    if (jlist == NULL) return;
    if (init_jlist(jlist, n) == FALSE) return;
    /* �ҳ��е�Ԫ�� */
    for (i = jlist->n; i > 0; i--) {
        s1 = ( s1 + m - 1 ) % i ;
        w = retrieve_seq(jlist, s1);     /* ���±�Ϊs1��Ԫ�ص�ֵ */
        printf("Out element %d \n", w);   /* Ԫ�س��� */
        delete_seq(jlist, s1);           /* ɾ�����е�Ԫ�� */
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

