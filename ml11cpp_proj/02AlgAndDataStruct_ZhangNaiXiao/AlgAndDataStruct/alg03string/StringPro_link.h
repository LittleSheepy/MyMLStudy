#pragma once
struct StrNode;						/* �����Ľڽ��*/
typedef struct StrNode* PStrNode;	/* ����ָ������*/
typedef struct StrNode* LinkString;	/* ����������*/
struct StrNode {
	char c;
	LinkString link;
};

void Str_link_test();
