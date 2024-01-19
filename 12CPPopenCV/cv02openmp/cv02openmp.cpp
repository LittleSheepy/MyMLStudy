// cv02openmp.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <windows.h>
#include <iostream>
#include <omp.h>
using namespace std;

int main()
{
    //int nProcessors = omp_get_max_threads();

    //std::cout << nProcessors << std::endl;
    //// 动态设定并行区域执行的线程
    omp_set_dynamic(1);

    ////设置是否允许OpenMP进行嵌套并行，默认的设置为false,此处为打开
    omp_set_nested(true);

    //omp_set_num_threads(nProcessors);

    std::cout << omp_get_num_threads() << std::endl;

//#pragma omp parallel for  
//    for (int i = 0; i < 5; i++) {
//        int tid = omp_get_thread_num();
//        std::cout << tid << "\t tid" << std::endl;
//        int nThreads = omp_get_num_threads();
//        std::cout << nThreads << "\t nThreads" << std::endl;
//    }
    DWORD begin, end, omp_time, time;
    begin = GetTickCount();
    std::cout << "Hello World!\n";
#pragma omp parallel for  //后面是for循环
    for (int i = 0; i < 100; i++)
    {
        int tid = omp_get_thread_num();
        cout << tid << "\t tid\n";
        //int nThreads = omp_get_num_threads();
        //std::cout << nThreads << "\t nThreads" << std::endl;
        int j = 100000 * 100;
        while (j-- > 0) {
            ;
        }
    }


    end = GetTickCount();
    omp_time = end - begin;
    cout << "omp time: " << omp_time << endl;


    begin = GetTickCount();
    for (int i = 0; i < 100; i++)
    {
        int j = 100000* 100;
        while (j-- > 0) {
            ;
        }
    }
    end = GetTickCount();
    time = end - begin;
    cout << "omp time: " << omp_time << endl;
    cout << "time: " << time << endl;

    return 0;
}

#define MAX_VALUE 100000000

double _test(int value)
{
    int index = 0;
    double result = 0.0;
    for (index = value + 1; index < MAX_VALUE; index += 2)
        result += 1.0 / index;

    return result;
}

void OpenMPTest()
{
    int index = 0;
    int time1 = 0;
    int time2 = 0;
    double value1 = 0.0, value2 = 0.0;
    double result[2];

    time1 = GetTickCount();
    for (index = 1; index < MAX_VALUE; index++)
        value1 += 1.0 / index;

    time1 = GetTickCount() - time1;
    memset(result, 0, sizeof(double) * 2);
    time2 = GetTickCount();

#pragma omp parallel for
    for (index = 0; index < 2; index++) {
        result[index] = _test(index);
        printf("index = %d, ThreadId = %d\n", index, omp_get_thread_num());
    }

    value2 = result[0] + result[1];
    time2 = GetTickCount() - time2;

    printf("time1 = %d,omp time2 = %d\n", time1, time2);
    return;
}

int main1()
{

    omp_set_dynamic(1);
    omp_set_num_threads(8);
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        printf("Hello from thread %d\n", thread_id);
    }

    OpenMPTest();

    system("pause");
    return 0;
}


// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
