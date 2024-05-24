#include "alg03string/string_seq.h"
//创建空字符串
PSeqString createNullStr_seq(int m) {
    PSeqString pstr = (PSeqString)malloc(sizeof(struct SeqString));
    if (pstr != NULL) {
        pstr->c = (char*)malloc(sizeof(char) * m);
        if (pstr->c) {
            pstr->n = 0;
            pstr->MAXUM = m;
            return pstr;
        }
        else {
            free(pstr);
        }
    }
    printf("Out of space!\n");
    return NULL;
}

//判断串是否为空串
int isNullList(PSeqString p) {
    return (p->n == 0);
}


//返回串的长度
int length(PSeqString p) {
    return (p->n);
}

//返回将串s1和串s2拼接在一起构成的新串
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

//在串s中,求从串的第i个字符开始连续j个字符所组成的字串
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

//如果串s2是s1的字串,则可求串s2在s1中第一次出现的位置
//带回溯的算法  非KMP   KMP算法待补充
int index1(PSeqString s1, PSeqString s2) {
    int i = 0, j = 0;
    while (i < s1->n && j < s2->n) {
        if (s1->c[i] == s2->c[j]) {
            i++;
            j++;
        }
        else {
            i = i - j + 1;
            j = 0;
        }
    }
    if (j >= s2->n) {
        return (i - j + 1);
    }
    else {
        return 0;
    }
}

//判断是否为回文字符串
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

//测试用例
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
    if (index1(s1, s2)) {
        printf("The s2 is s1's substring! 第一次出现在字符串s1的第%d个位置\n", index1(s1, s2));
    }
    printf("s1和s2的组合是%s\n", concat(s1, s2)->c);
    printf("s3和s4的组合是%s\n", concat(s3, s4)->c);
    printf("Test the subStr function: %s\n", subStr(s1, 11, 5)->c);
    if (isPalindromeStr(s5)) {
        printf("s5是回文字符串\n");
    }
}

