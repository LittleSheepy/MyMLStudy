/* 线性表的 单链表 表示：函数实现*/

#include <stdio.h>
#include <stdlib.h>

#include "llist.h"

/* 创建一个带头结点的空链表 */
LinkList  createNullList_link( void ) {   
    LinkList llist;
    llist = (LinkList)malloc( sizeof( struct Node ) );	/* 申请表头结点空间 */
    if( llist != NULL ) llist->link = NULL;
    return llist;
}

/* 在llist带头结点的单链表中下标为i的(第i+1个)结点前插入元素x */
int insert_link(LinkList llist, int i, DataType x) { 
    PNode p = llist, q;
    int j;
    for (j = 0 ; p != NULL && j < i; j++)		/* 找下标为i-1的(第i个)结点 */
        p = p->link;
 	  
    if (j != i) {								/* i<1或者大于表长 */
        printf("Index of link-list is out of range.\n",i);
  	 	return 0;
    }
 	  
    q = (PNode)malloc( sizeof( struct Node ) );	/* 申请新结点 */
    if( q == NULL ) { 
        printf( "Out of space!\n" );
        return 0;
    }
    									/* 插入链表中 */
    q->info = x;
    q->link = p->link;
    p->link = q;						/* 注意该句必须在上句后执行 */
    return 1 ;
}

/* 在llist带有头结点的单链表中删除第一个值为x的结点 */
/* 这时要求 DataType 可以用 != 比较 */
int delete_link( LinkList llist, DataType x ) { 
    PNode p = llist, q;      	
    /*找值为x的结点的前驱结点的存储位置 */
    while( p->link != NULL && p->link->info != x )
        p = p->link;
       	
    if( p->link == NULL ) {  	/* 没找到值为x的结点 */
        printf("Datum does not exist!\n ");
        return 0;
    }
    
    q = p->link;	  			/* 找到值为x的结点 */
    p->link = p->link->link;  	/* 删除该结点 */
    free( q );      
    return 1; 
}

/* 在llist带有头结点的单链表中找第一个值为x的结点存储位置 */
/* 找不到时返回空指针 */
PNode locate_link( LinkList llist, DataType x ) {
    PNode p;
    if (llist == NULL)  return NULL;
    
    for ( p = llist->link; p != NULL && p->info != x; )
        p = p->link;
    return p;
}

/* 在带有头结点的单链表llist中下标为i的(第i+1个)结点的存储位置 */
/* 当表中无下标为i的(第i+1个)元素时，返回值为NULL */
PNode find_link( LinkList llist, int i ) { 
    PNode p;
    int j;
    if (i < 0) {					/* 检查i的值 */
        printf("Index of link-list is out of range.\n",i);
        return NULL;
    }
	  
    for ( p = llist->link, j = 0; p != NULL && j < i; j++) 
        p = p->link;

    if (p == NULL) 
        printf("Index of link-list is out of range.\n", i);

    return p;
}

/* 判断llist带有头结点的单链表是否是空链表 */
int  isNullList_link( LinkList llist) {     
    return llist == NULL || llist->link == NULL;
}

