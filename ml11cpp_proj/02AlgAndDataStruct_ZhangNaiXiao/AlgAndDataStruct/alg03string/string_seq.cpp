#include "alg03string/string_seq.h"
//�������ַ���
PSeqString createNullStr_seq(int m) {
    PSeqString pstr = (PSeqString)malloc(sizeof(struct SeqString));
    if (pstr != NULL) {
        pstr->c = (char*)malloc(sizeof(char) * m);
        if (pstr->c) {
            pstr->c[0] = 0;
            pstr->n = 0;
            pstr->MAXUM = m;
            return pstr;
        }
        else {
            free(pstr);
        }
    }
    printf("Out of space!!\n");
    return NULL;
}

//�жϴ��Ƿ�Ϊ�մ�
int isNullList(PSeqString p) {
    return (p->n == 0);
}


//���ش��ĳ���
int length(PSeqString p) {
    return (p->n);
}

//���ؽ���s1�ʹ�s2ƴ����һ�𹹳ɵ��´�
PSeqString concat(PSeqString s1, PSeqString s2) {
    PSeqString s = createNullStr_seq(MAX * 2);
    s->n = s1->n + s2->n;
    int i;
    for (i = 0; i < s1->n; i++) {
        s->c[i] = s1->c[i];
    }
    for (i = s1->n; i < s->n; i++) {
        s->c[i] = s2->c[i - s1->n];
    }
    return s;
}
PSeqString subStr_seq(PSeqString s, int i, int j) {
    PSeqString s1;
    int k;
    s1 = createNullStr_seq(j);
    if (s1 == NULL)return NULL;
    if (i > 0 && i <= s->n && j > 0) {
        if (s->n < i + j - 1)j = s->n - i + 1;
        /*����i��ʼȡ����j���ַ�������ȡ������ȡ����*/
        for (k = 0; k < j; k++)
            s1->c[k] = s->c[i + k - 1]; /*���ִ���ֵ*/
        s1->n = j;
    }
    return s1;
}
//�ڴ�s��,��Ӵ��ĵ�i���ַ���ʼ����j���ַ�����ɵ��ִ�
PSeqString subStr(PSeqString s, int i, int j) {
    if (i > s->n || i + j - 1 > s->n) {
        printf("Out of space!\n");
        return NULL;
    }
    if (j <= 0) {
        printf("Wrong inpput! Please make the third parameter greater(it must greater than 0)!\n");
        return NULL;
    }
    PSeqString p = createNullStr_seq(MAX);
    s->n = j;
    int k;
    for (k = i - 1; k < i + j - 1; k++) {
        p->c[k - i + 1] = s->c[k];
    }
    return p;
}

//�����s2��s1���ִ�,�����s2��s1�е�һ�γ��ֵ�λ��
//�����ݵ��㷨  ��KMP   KMP�㷨������
int index(PSeqString t, PSeqString p) {
    int i, j;
    i = 0; j = 0;
    while (i < p->n && j < t->n) {
        if (p->c[i] == t->c[j]) {
            i++; j++;
        }
        else {
            j = j - i + 1;
            i = 0;
        }
    }
    if (i >= p->n)
        return(j - p->n + 1);
    else return 0;
}

//�ж��Ƿ�Ϊ�����ַ���
int isPalindromeStr(PSeqString p) {
    int i, result;
    result = (p->n) / 2;
    for (i = 0; i < result; i++) {
        if (p->c[i] != p->c[p->n - i - 1]) {
            break;
        }
    }
    if (i == result) {
        return 1;
    }
    else {
        return 0;
    }
}

//��������
void string_test() {
    PSeqString s1, s2, s3, s4, s5;
    s1 = createNullStr_seq(MAX);
    s2 = createNullStr_seq(MAX);
    s3 = createNullStr_seq(MAX);
    s4 = createNullStr_seq(MAX);
    s5 = createNullStr_seq(MAX);
    s1->c = (char*)"hahahahahajwhanandyou";
    s1->n = 21;
    s2->c = (char*)"jwhan";
    s2->n = 5;
    s3->c = (char*)"";
    s4->c = (char*)"";
    s3->n = 0;
    s4->n = 0;
    s5->c = (char*)"hahauahah";
    s5->n = 9;
    printf("The String s1's length is %d\n", length(s1));
    printf("The String s2's length is %d\n", length(s2));
    if (index(s1, s2)) {
        printf("The s2 is s1's substring! ��һ�γ������ַ���s1�ĵ�%d��λ��\n", index(s1, s2));
    }
    printf("s1��s2�������%s\n", concat(s1, s2)->c);
    printf("s3��s4�������%s\n", concat(s3, s4)->c);
    printf("Test the subStr function��%s 11 5: %s\n", s1->c, subStr_seq(s1, 11, 5)->c);
    if (isPalindromeStr(s5)) {
        printf("s5�ǻ����ַ���\n");
    }
}

