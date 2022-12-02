
// tensorRT include
// 编译用的头文件
#include <NvInfer.h>

// onnx解析器的头文件
#include <onnx-tensorrt/NvOnnxParser.h>

// 推理用的运行时头文件
#include <NvInferRuntime.h>

// cuda include
#include <cuda_runtime.h>

// system include
#include <stdio.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#ifdef WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <opencv2/opencv.hpp>

#include "common.h"
using namespace std;

static vector<int> _classes_colors = {
    0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 
    128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 
    64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 12
};
// 通过智能指针管理nv返回的指针参数
// 内存自动释放，避免泄漏
template<typename _T>
shared_ptr<_T> make_nvshared(_T* ptr){
    return shared_ptr<_T>(ptr, [](_T* p){p->destroy();});
}



vector<string> load_lines(const char* file){
    vector<string> lines;

    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open()){
        printf("open %s failed.\n", file);
        return lines;
    }
    
    string line;
    while(getline(in, line)){
        lines.push_back(line);
    }
    in.close();
    return lines;
}

unordered_map<string, int> load_vocab(const string& file, bool add_lower_case=true){

    unordered_map<string, int> vocab;
    auto lines = load_lines(file.c_str());
    for(int i = 0; i < lines.size(); ++i){
        auto token = lines[i];
        vocab[token] = i;

        if(add_lower_case){
            if(!token.empty() && token[0] != '[' && token.back() != ']'){
                for(int j = 0; j < token.size(); ++j){
                    if(token[j] >= 'A' && token[j] <= 'Z')
                        token[j] = token[j] - 'A' + 'a';
                }
            }
            if(vocab.find(token) == vocab.end())
                vocab[token] = i;
        }
    }
    return vocab;
}

int find_token(const string& token, const unordered_map<string, int>& vocab){
    auto iter = vocab.find(token);
    if(iter == vocab.end()){
        if(!token.empty() && token[0] != '[' && token.back() != ']'){
            string new_token = token;
            bool has_upper = false;
            for(int j = 0; j < new_token.size(); ++j){
                if(new_token[j] >= 'A' && new_token[j] <= 'Z'){
                    new_token[j] = new_token[j] - 'A' + 'a';
                    has_upper = true;
                }
            }

            if(has_upper){
                iter = vocab.find(new_token);
                if(iter != vocab.end())
                    return iter->second;
            }
        }
        return -1;
    }
    return iter->second;
}

/* utf-8
  拆分utf8的汉字，把汉字部分独立，ascii部分连续为一个
  for example:
    你jok我good呀  -> ["你", "job", "我", "good", "呀"] */
tuple<vector<string>, vector<tuple<int, int>>> split_words(const string& text){

    // 1字节：0xxxxxxx 
    // 2字节：110xxxxx 10xxxxxx 
    // 3字节：1110xxxx 10xxxxxx 10xxxxxx 
    // 4字节：11110xxx 10xxxxxx 10xxxxxx 10xxxxxx 
    // 5字节：111110xx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx
    // 6字节：11111110 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx
    unsigned char* up = (unsigned char*)text.c_str();
    int offset = 0;
    int length = text.size();
    unsigned char lab_char[] = {
    // 11111110  11111000  11110000  11100000  11000000  01111111
        0xFE,    0xF8,     0xF0,     0xE0,     0xC0,     0x80
    };

    int char_size_table[] = {
        6, 5, 4, 3, 2, 0
    };

    vector<tuple<int, int>> offset_map;
    vector<string> tokens;
    string ascii;
    int state = 0;   // 0 none,  1 wait ascii
    int ascii_start = 0;
    while(offset < length){
        unsigned char token = up[offset];
        int char_size = 1;
        for(int i = 0; i < 6; ++i){
            if(token >= lab_char[i]){
                char_size = char_size_table[i];
                break;
            }
        }

        if(char_size == 0){
            // invalid char
            offset++;
            continue;
        }

        if(offset + char_size > length)
            break;

        auto char_string = text.substr(offset, char_size);
        if(char_size == 1 && token != ' '){
            // ascii 
            if(state == 0){
                ascii = char_string;
                ascii_start = offset;
                state = 1;
            }else if(state == 1){
                ascii += char_string;
            }
        }else{
            if(state == 1){
                tokens.emplace_back(ascii);
                offset_map.emplace_back(ascii_start, offset);
                state = 0;
            }

            if(token != ' '){
                offset_map.emplace_back(offset, offset+char_size);
                tokens.emplace_back(char_string);
            }
        }
        offset += char_size;
    }

    if(state == 1){
        tokens.emplace_back(ascii);
        offset_map.emplace_back(ascii_start, offset);
    }
    return make_tuple(tokens, offset_map);
}

/* 把字符串拆分为词组，汉字单个为一组 */
tuple<vector<string>, vector<tuple<int, int>>> tokenize(const string& text, const unordered_map<string, int>& vocab, int max_length, bool case_to_lower=false){

    vector<tuple<int, int>> offset_map;
    vector<tuple<int, int>> offset_newmap;
    vector<string> tokens;
    vector<string> output;
    auto UNK = "[UNK]";
    tie(tokens, offset_map) = split_words(text);

    for(int itoken = 0; itoken < tokens.size(); ++itoken){
        auto& chars = tokens[itoken];
        int char_start = 0;
        int char_end = 0;
        tie(char_start, char_end) = offset_map[itoken];

        if(chars.size() > max_length){
            output.push_back(UNK);
            offset_newmap.emplace_back(char_start, char_end);
            continue;
        }

        bool is_bad = false;
        int start = 0;
        vector<string> sub_tokens;
        vector<tuple<int, int>> sub_offsetmap;
        while(start < chars.size()){
            int end = chars.size();
            string cur_substr;
            while(start < end){
                auto substr = chars.substr(start, end-start);

                if(case_to_lower){
                    for(int k = 0; k < substr.size(); ++k){
                        auto& c = substr[k];
                        if(c >= 'A' && c <= 'Z')
                            c = c - 'A' + 'a';
                    }
                }

                if(start > 0)
                    substr = "##" + substr;

                auto token_id = find_token(substr, vocab);
                if(token_id != -1){
                    cur_substr = substr;
                    break;
                }
                end -= 1;
            }

            if(cur_substr.empty()){
                is_bad = true;
                break;
            }
            sub_tokens.push_back(cur_substr);
            sub_offsetmap.emplace_back(char_start + start, char_start + end);
            start = end;
        }

        if(is_bad){
            output.push_back(UNK);
            offset_newmap.emplace_back(char_start, char_end);
        }else{
            output.insert(output.end(), sub_tokens.begin(), sub_tokens.end());
            offset_newmap.insert(offset_newmap.end(), sub_offsetmap.begin(), sub_offsetmap.end());
        }
    }
    return make_tuple(output, offset_newmap);
}

vector<int> tokens_to_ids(const vector<string>& tokens, const unordered_map<string, int>& vocab){
    vector<int> output(tokens.size());
    for(int i =0 ; i < tokens.size(); ++i)
        output[i] = find_token(tokens[i], vocab);
    return output;
}

tuple<vector<int>, vector<int>, vector<tuple<int, int>>, int> align_and_pad(
    const vector<string>& tokens, vector<tuple<int, int>>& offset_map, int pad_size, 
    const unordered_map<string, int>& vocab
){
    auto CLS = find_token("[CLS]", vocab);
    auto SEP = find_token("[SEP]", vocab);
    vector<int> output = tokens_to_ids(tokens, vocab);
    vector<int> mask(pad_size, 1);
    output.insert(output.begin(), CLS);
    output.insert(output.end(), SEP);
    offset_map.insert(offset_map.begin(), make_tuple(0, 0));
    offset_map.insert(offset_map.end(), make_tuple(0, 0));

    int old_size = output.size();
    output.resize(pad_size);

    if(old_size < pad_size){
        std::fill(output.begin() + old_size, output.end(),   0);
        std::fill(mask.begin()   + old_size, mask.end(),     0);
    }else{
        output.back() = SEP;
        offset_map.back() = make_tuple(0, 0);
    }
    return make_tuple(output, mask, offset_map, old_size);
}

// input_ids, attention_mask, offset_map, word_length
tuple<vector<int>, vector<int>, vector<tuple<int, int>>, int> make_text_data(const string& text, const unordered_map<string, int>& vocab, int max_length){

    vector<string> tokens;
    vector<tuple<int, int>> offset_map;
    tie(tokens, offset_map) = tokenize(text, vocab, max_length);
    return align_and_pad(tokens, offset_map, max_length, vocab);
}


// 上一节的代码
bool build_model16(){

    if(exists("classifier.trtmodel")){
        printf("classifier.trtmodel has exists.\n");
        return true;
    }

    TRTLogger logger;

    // 这是基本需要的组件
    auto builder = make_nvshared(nvinfer1::createInferBuilder(logger));
    auto config = make_nvshared(builder->createBuilderConfig());

    // createNetworkV2(1)表示采用显性batch size，新版tensorRT(>=7.0)时，不建议采用0非显性batch size
    // 因此贯穿以后，请都采用createNetworkV2(1)而非createNetworkV2(0)或者createNetwork
    auto network = make_nvshared(builder->createNetworkV2(1));

    // 通过onnxparser解析器解析的结果会填充到network中，类似addConv的方式添加进去
    auto parser = make_nvshared(nvonnxparser::createParser(*network, logger));
    if(!parser->parseFromFile("classifier.onnx", 1)){
        printf("Failed to parse classifier.onnx\n");

        // 注意这里的几个指针还没有释放，是有内存泄漏的，后面考虑更优雅的解决
        return false;
    }
    
    int maxBatchSize = 1;
    printf("Workspace Size = %.2f MB\n", (1 << 30) / 1024.0f / 1024.0f);
    config->setMaxWorkspaceSize(1 << 30);

    // 如果模型有多个输入，则必须多个profile
    auto profile = builder->createOptimizationProfile();
    for(int i = 0; i < network->getNbInputs(); ++i){
        auto input_tensor = network->getInput(i);
        auto input_dims = input_tensor->getDimensions();
        
        // 配置最小允许batch
        input_dims.d[0] = 1;
        input_dims.d[1] = 32;   // seq length
        profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
        profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
        input_dims.d[0] = maxBatchSize;
        profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
    }
    config->addOptimizationProfile(profile);

    auto engine = make_nvshared(builder->buildEngineWithConfig(*network, *config));
    if(engine == nullptr){
        printf("Build engine failed.\n");
        return false;
    }

    // 将模型序列化，并储存为文件
    auto model_data = make_nvshared(engine->serialize());
    FILE* f = fopen("classifier.trtmodel", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    // 卸载顺序按照构建顺序倒序
    printf("Build Done.\n");
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////


void inference16(){

    TRTLogger logger;
    auto engine_data = load_file("classifier.trtmodel");
    auto runtime   = make_nvshared(nvinfer1::createInferRuntime(logger));
    auto engine = make_nvshared(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return;
    }

    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));
    auto execution_context = make_nvshared(engine->createExecutionContext());

    int input_batch = 1;
    int input_seqlength = 32;
    int input_numel = input_batch * input_seqlength;
    auto vocab = load_vocab("../Bert-Chinese-Text-Classification/bert_pretrain/vocab.txt");
    int* input_ids_device = nullptr;
    int* input_mask_device = nullptr;
    checkRuntime(cudaMalloc(&input_ids_device, input_numel * sizeof(int)));
    checkRuntime(cudaMalloc(&input_mask_device, input_numel * sizeof(int)));

    ///////////////////////////////////////////////////
    // letter box
    const char* input = "2岁男童爬窗台不慎7楼坠下获救(图)";
    vector<int> input_ids, attention_mask;
    int sentence_length = 0;
    vector<tuple<int, int>> offset_map;
    tie(input_ids, attention_mask, offset_map, sentence_length) = make_text_data(input, vocab, input_seqlength);

    ///////////////////////////////////////////////////
    checkRuntime(cudaMemcpyAsync(input_ids_device, input_ids.data(), input_numel * sizeof(int), cudaMemcpyHostToDevice, stream));
    checkRuntime(cudaMemcpyAsync(input_mask_device, attention_mask.data(), input_numel * sizeof(int), cudaMemcpyHostToDevice, stream));

    // 3x3输入，对应3x3输出
    auto output_dims  = engine->getBindingDimensions(2);
    int num_classes   = output_dims.d[1];
    int output_numel  = input_batch * num_classes;
    float* output_data_host = nullptr;
    float* output_data_device = nullptr;
    checkRuntime(cudaMallocHost(&output_data_host, sizeof(float) * output_numel));
    checkRuntime(cudaMalloc(&output_data_device, sizeof(float) * output_numel));

    // 明确当前推理时，使用的数据输入大小
    auto input_ids_dims = engine->getBindingDimensions(0);
    input_ids_dims.d[0] = input_batch;
    input_ids_dims.d[1] = input_seqlength;
    execution_context->setBindingDimensions(0, input_ids_dims);

    auto input_mask_dims = engine->getBindingDimensions(1);
    input_mask_dims.d[0] = input_batch;
    input_mask_dims.d[1] = input_seqlength;
    execution_context->setBindingDimensions(1, input_mask_dims);

    void* bindings[] = {input_ids_device, input_mask_device, output_data_device};
    bool success      = execution_context->enqueueV2(bindings, stream, nullptr);
    checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device, sizeof(float) * output_numel, cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));

    const char* class_label[] = {
        "金融", // finance
        "房地产", // realty
        "股票", // stocks
        "教育", // education
        "科学", // science
        "社会", // society
        "政治", // politics
        "体育", // sports
        "游戏", // game
        "娱乐"  // entertainment
    };

    int predict_label = std::max_element(output_data_host, output_data_host + num_classes) - output_data_host;
    float confidence = output_data_host[predict_label];
    printf("Input: %s\n", input);
    printf("Predict: %s  %f\n", class_label[predict_label], confidence);
    
    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFreeHost(output_data_host));
    checkRuntime(cudaFree(output_data_device));
}

int chinese_classifer_bert(){
    if(!build_model16()){
        return -1;
    }
    inference16();
    return 0;
}