#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include "StringPro_link.h"

// 创建带头结点的空链串
LinkString createNullStr_link(void) {
	LinkString pst;
	pst = (LinkString)malloc(sizeof(struct StrNode));
	if (pst != NULL)
		pst->link = NULL;
	else printf("Out of space!\n");
	return pst;
}

//检测是否为空字符串    带头节点
int isNullStr_link(PStrNode s) {
	return (s->link == NULL);
}

//赋值处理
int assign(LinkString s, const char* a) {
	int i = 0;
	PStrNode tmp=s;
	while (1) {
		if (a[i] == '\0') {
			break;
		}
		PStrNode p = createNullStr_link();
		p->c = a[i];
		tmp->link = p;
		tmp = p;
		i++;
	}
	return 1;
}

//返回字符串的长度      带头结点
int length(LinkString s) {
	int counter = 0;
	PStrNode p;
	p = s->link;
	while (p != NULL) {
		counter++;
		p = p->link;
	}

	return counter;
}

/* 求从s所指的头结点的链串中第i(i>0)个字符开始连续取j个字符所构成的子串 */
LinkString subStr_link(LinkString s, int i, int j) {
	LinkString s1;
	PStrNode p, q, t;
	int k;
	s1 = createNullStr_link();			/* 创建空链串 */
	if (s1 == NULL) {
		printf("Out of space!\n");
		return NULL;
	}
	if (i < 1 || j < 1) return s1;		/* i,j值不合适，返回空串*/
	p = s;
	for (k = 1; k <= i; k++)			/*找第i个结点*/
		if (p != NULL)
			p = p->link;
		else
			return s1;
	if (p == NULL) return s1;
	t = s1;
	for (k = 1; k <= j; k++) /*连续取j个字符*/
		if (p != NULL) {
			q = (PStrNode)malloc(sizeof(struct StrNode));
			if (q == NULL) {
				printf("Out of space!\n");
				return s1;
			}
			q->c = p->c;
			q->link = NULL;
			t->link = q; /*结点放入子链串中 */
			t = q;
			p = p->link;
		}
	return s1;
}
//返回将串m和串n拼接在一起的新串    带头结点
LinkString concat(LinkString m, LinkString n) {
	PStrNode result = createNullStr_link();
	result->link = m->link;

	PStrNode p, q, t;
	p = result;
	while (p->link != NULL) {
		p = p->link;
	}
	p->link = n->link;
	return result;
}
//打印字符串    带头结点
int printStr(PStrNode s) {
	if (s == NULL) {
		printf("数据不合法,请检查!\n");
		return 0;
	}
	while (s->link != NULL) {
		printf("%c", s->link->c);
		s = s->link;
	}
	//printf("%c\n", s->link->c);
	printf("\n");
	return 1;
}
void Str_link_test() {
	LinkString lst1, lst2, lst3;
	lst1 = createNullStr_link();
	assign(lst1, "ABCDEF");
	printStr(lst1);
	printf("lst1's length is %d.\n", length(lst1));
	lst2 = subStr_link(lst1, 3, 2);
	printStr(lst2);
	lst3 = concat(lst1, lst2);
	printStr(lst3);
}