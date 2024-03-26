#include <stdio.h>
#include "alg02list/slist_test.h"

void print_list(PSeqList palist) {
    printf("%s", "打印列表:\n");
    if (1 == isNullList_seq(palist)) {
        printf("%s", "列表是空的！\n");
    }
    else {
        printf("%s\n", "列表是非空的！");
        printf("n=%d\n", palist->n);
        for (int i = 0; i < palist->n; i++)
        {
            printf("element[%d]=%d\n", i, palist->element[i]);
        }
    }
}

void print_pklist(LinkList palist) {
    printf("%s", "打印列表:\n");
    if (1 == isNullList_link(palist)) {
        printf("%s", "列表是空的！\n");
    }
    else {
        printf("%s\n", "列表是非空的！");
        while (palist->link != NULL)
        {
            palist = palist->link;
            printf("element=%d\n", palist->info);
        }
    }
}

void slist_test(void) {
    PSeqList palist = createNullList_seq(10);
    print_list(palist);

    insert_seq(palist, 0, 11);
    insert_seq(palist, 1, 12);
    print_list(palist);
    int loc = locate_seq(palist, 12);
    printf("loc = %d\n", loc);
    DataType data = retrieve_seq(palist, 0);
    printf("data = %d\n", data);

    delete_seq(palist, 0);
    print_list(palist);
    loc = locate_seq(palist, 12);
    printf("loc = %d\n", loc);
}

void dslist_test(void) {

}

void lklist_test(void) {
    LinkList lklist = createNullList_link();
    print_pklist(lklist);

    insert_link(lklist, 0, 1);
    insert_link(lklist, 1, 2);
    insert_link(lklist, 2, 3);
    insert_link(lklist, 3, 4);
    print_pklist(lklist);

    delete_link(lklist, 2);
    print_pklist(lklist);

    PNode p1 = locate_link(lklist, 3);
    printf("locate_link info=3, p1=%d\n", p1->info);

    PNode p2 = find_link(lklist, 2);
    printf("find_link pos=2, p2=%d\n", p2->info);

}