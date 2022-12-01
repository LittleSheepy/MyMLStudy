#include "common.h"
#include <stdio.h>


#include <iostream>
#include <string>
#include <locale>
#include <codecvt>
#include <fstream>
#include <codecvt>
#include <array>
#include <xlocbuf>
#include <Shlwapi.h>
#pragma comment(lib, "Shlwapi.lib")
bool exists(const string& path) {

#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}
bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line) {
    if (code != cudaSuccess) {
        const char* err_name = cudaGetErrorName(code);
        const char* err_message = cudaGetErrorString(code);
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
        return false;
    }
    return true;
}
nvinfer1::Weights make_weights(float* ptr, int n) {
    nvinfer1::Weights w;
    w.count = n;
    w.type = nvinfer1::DataType::kFLOAT;
    w.values = ptr;
    return w;
}

vector<unsigned char> load_file(const string& file) {
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0) {
        in.seekg(0, ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}
inline const char* severity_string(nvinfer1::ILogger::Severity t) {
    switch (t) {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
    case nvinfer1::ILogger::Severity::kERROR:   return "error";
    case nvinfer1::ILogger::Severity::kWARNING: return "warning";
    case nvinfer1::ILogger::Severity::kINFO:    return "info";
    case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
    default: return "unknow";
    }
}
void TRTLogger::log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept {
    if (severity <= Severity::kINFO) {
        // 打印带颜色的字符，格式如下：
        // printf("\033[47;33m打印的文本\033[0m");
        // 其中 \033[ 是起始标记
        //      47    是背景颜色
        //      ;     分隔符
        //      33    文字颜色
        //      m     开始标记结束
        //      \033[0m 是终止标记
        // 其中背景颜色或者文字颜色可不写
        // 部分颜色代码 https://blog.csdn.net/ericbar/article/details/79652086
        if (severity == Severity::kWARNING) {
            printf("===\033[33m>>>>%s: %s\033[0m\n", severity_string(severity), msg);
        }
        else if (severity <= Severity::kERROR) {
            printf("===\033[31m>>>>%s: %s\033[0m\n", severity_string(severity), msg);
        }
        else {
            printf("===>>>>%s: %s\n", severity_string(severity), msg);
        }
    }
}

string to_string(const wstring& str, const locale& loc = locale())
{
    vector<char>buf(str.size());
    use_facet<ctype<wchar_t>>(loc).narrow(str.data(), str.data() + str.size(), '*', buf.data());
    return string(buf.data(), buf.size());
}
#include <Windows.h>
//将wstring转换成string  
string wstring2string(wstring wstr)
{
    string result;
    //获取缓冲区大小，并申请空间，缓冲区大小事按字节计算的  
    int len = WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), wstr.size(), NULL, 0, NULL, NULL);
    char* buffer = new char[len + 1];
    //宽字节编码转换成多字节编码  
    WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), wstr.size(), buffer, len, NULL, NULL);
    buffer[len] = '\0';
    //删除缓冲区并返回值  
    result.append(buffer);
    delete[] buffer;
    return result;
}

std::string UnicodeToAscii(const std::wstring str)
{
    int	iTextLen = WideCharToMultiByte(CP_ACP, 0, str.c_str(), -1, NULL, 0, NULL, NULL);
    std::vector<char> vecText(iTextLen, '\0');
    ::WideCharToMultiByte(CP_ACP, 0, str.c_str(), -1, &(vecText[0]), iTextLen, NULL, NULL);

    std::string strText = &(vecText[0]);

    return strText;
}
std::string UTF8ToString(const std::string& utf8Data)
{
    //先将UTF-8转换成Unicode
    std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
    std::wstring wString = conv.from_bytes(utf8Data);
    //在转换成string
    //return wstring2string(wString);
    return UnicodeToAscii(wString);
    //return wString;
}

constexpr char const* rc = "?"; // replacement_char
// table mapping ISO-8859-1 characters to similar ASCII characters
std::array<char const*, 96> conversions = { {
   " ",  "!","c","L", rc,"Y", "|","S", rc,"C","a","<<",   rc,  "-",  "R", "-",
    rc,"+/-","2","3","'","u", "P",".",",","1","o",">>","1/4","1/2","3/4", "?",
   "A",  "A","A","A","A","A","AE","C","E","E","E", "E",  "I",  "I",  "I", "I",
   "D",  "N","O","O","O","O", "O","*","0","U","U", "U",  "U",  "Y",  "P","ss",
   "a",  "a","a","a","a","a","ae","c","e","e","e", "e",  "i",  "i",  "i", "i",
   "d",  "n","o","o","o","o", "o","/","0","u","u", "u",  "u",  "y",  "p", "y"
} };
template <class Facet>
class usable_facet : public Facet {
public:
    using Facet::Facet;
    ~usable_facet() {}
};
std::string to_ascii(std::string const& utf8) {
    std::wstring_convert<usable_facet<std::codecvt<char32_t, char, std::mbstate_t>>,
        char32_t> convert;
    std::u32string utf32 = convert.from_bytes(utf8);

    std::string ascii;
    for (char32_t c : utf32) {
        if (c <= U'\u007F')
            ascii.push_back(static_cast<char>(c));
        else if (U'\u00A0' <= c && c <= U'\u00FF')
            ascii.append(conversions[c - U'\u00A0']);
        else
            ascii.append(rc);
    }
    return ascii;
}

vector<string> load_labels(const char* file) {
    vector<string> lines;

    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open()) {
        printf("open %d failed.\n", file);
        return lines;
    }

    string line;
    while (getline(in, line)) {
        lines.push_back(line);
    }
    in.close();
    return lines;
}