#pragma once
struct StrNode;						/* 链串的节结点*/
typedef struct StrNode* PStrNode;	/* 结点的指针类型*/
typedef struct StrNode* LinkString;	/* 链串的类型*/
struct StrNode {
	char c;
	LinkString link;
};

void Str_link_test();
