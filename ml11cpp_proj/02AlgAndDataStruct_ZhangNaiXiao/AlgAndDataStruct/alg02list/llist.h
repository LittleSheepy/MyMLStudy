/* 线性表的单链表表示：类型和界面函数定义*/

/* 定义链接表元素类型。应根据需要定义 */
typedef int DataType;

struct Node;					  /* 单链表结点类型 */
typedef  struct Node  *PNode;	  /* 结点指针类型 */
typedef  struct Node  *LinkList;  /* 单链表类型 */

struct  Node { 					  /* 单链表结点结构 */
    DataType  info;
    PNode     link;
};

/* 创建一个带头结点的空链表 */
LinkList createNullList_link( void );

/* 在llist带头结点的单链表中下标为i的(第i+1个)结点前插入元素x */
int insert_link(LinkList llist, int i, DataType x);

/* 在llist带有头结点的单链表中删除第一个值为x的结点 */
int  delete_link( LinkList llist, DataType x );

/* 在llist带有头结点的单链表中找第一个值为x的结点存储位置 */
PNode  locate_link(LinkList llist, DataType x );

/* 在带有头结点的单链表llist中求下标为i的(第i+1个)结点的存储位置 */
/* 当表中无下标为i的(第i+1个)元素时，返回值为NULL */
PNode  find_link( LinkList llist, int i );

/* 判断llist带有头结点的单链表是否是空链表 */
int  isNullList_link( LinkList llist);
