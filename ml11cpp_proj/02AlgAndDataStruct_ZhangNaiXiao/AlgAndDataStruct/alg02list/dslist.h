/* 线性表的顺序表示：类型和界面定义*/

#define TRUE 1
#define FALSE 0
#define	SPECIAL -1

/* 定义顺序表的大小。应根据需要修改 */
enum { NBASE = 20 };

/* 定义顺序表的元素类型。应根据需要修改 */
typedef int DataType;

struct SeqList {  	
    int n, nmax;			/* 当前元素个数 n，存储区大小 nmax */
    DataType *elems;	/* 存放线性表中的元素 */
};

typedef struct SeqList *PSeqList;

/* 创建新的顺序表 */
PSeqList createNullList_seq( void );

/* 判断顺序表是否为空 */
int isNullList_seq( PSeqList palist );

/*在palist所指顺序表中下标为p的元素之前插入元素x*/
int insert_seq(PSeqList palist, int p, DataType x);

/*在palist所指顺序表中删除下标为ｐ的元素*/
int  delete_seq( PSeqList palist, int p );

/*求x在palist所指顺序表中的下标*/
int locate_seq(PSeqList palist, DataType x);

/* 求palist所指顺序表中下标为p的元素值 */
DataType  retrieve_seq( PSeqList palist, int p );
