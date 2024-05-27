#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include "StringPro_link.h"

// ������ͷ���Ŀ�����
LinkString createNullStr_link(void) {
	LinkString pst;
	pst = (LinkString)malloc(sizeof(struct StrNode));
	if (pst != NULL)
		pst->link = NULL;
	else printf("Out of space!\n");
	return pst;
}

//����Ƿ�Ϊ���ַ���    ��ͷ�ڵ�
int isNullStr_link(PStrNode s) {
	return (s->link == NULL);
}

//��ֵ����
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

//�����ַ����ĳ���      ��ͷ���
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

/* ���s��ָ��ͷ���������е�i(i>0)���ַ���ʼ����ȡj���ַ������ɵ��Ӵ� */
LinkString subStr_link(LinkString s, int i, int j) {
	LinkString s1;
	PStrNode p, q, t;
	int k;
	s1 = createNullStr_link();			/* ���������� */
	if (s1 == NULL) {
		printf("Out of space!\n");
		return NULL;
	}
	if (i < 1 || j < 1) return s1;		/* i,jֵ�����ʣ����ؿմ�*/
	p = s;
	for (k = 1; k <= i; k++)			/*�ҵ�i�����*/
		if (p != NULL)
			p = p->link;
		else
			return s1;
	if (p == NULL) return s1;
	t = s1;
	for (k = 1; k <= j; k++) /*����ȡj���ַ�*/
		if (p != NULL) {
			q = (PStrNode)malloc(sizeof(struct StrNode));
			if (q == NULL) {
				printf("Out of space!\n");
				return s1;
			}
			q->c = p->c;
			q->link = NULL;
			t->link = q; /*�������������� */
			t = q;
			p = p->link;
		}
	return s1;
}
//���ؽ���m�ʹ�nƴ����һ����´�    ��ͷ���
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
//��ӡ�ַ���    ��ͷ���
int printStr(PStrNode s) {
	if (s == NULL) {
		printf("���ݲ��Ϸ�,����!\n");
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