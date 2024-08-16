
def load_model_modelscope():
    from modelscope import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    # Note: The default behavior now has injection attack prevention off.
    tokenizer = AutoTokenizer.from_pretrained(qwen_path, revision='master', trust_remote_code=True)

    # use bf16
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-1_8B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
    # use fp16
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-1_8B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
    # use cpu only
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-1_8B-Chat", device_map="cpu", trust_remote_code=True).eval()
    # use auto mode, automatically select precision based on the device.
    model = AutoModelForCausalLM.from_pretrained(qwen_path, revision='master', device_map="auto", trust_remote_code=True).eval()

    # Specify hyperparameters for generation. But if you use transformers>=4.32.0, there is no need to do this.
    # model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-1_8B-Chat", trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参

    return model, tokenizer


def load_model_transformers():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.generation import GenerationConfig

    # 可选的模型包括: "Qwen/Qwen-7B-Chat", "Qwen/Qwen-14B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)

    # 打开bf16精度，A100、H100、RTX3060、RTX3070等显卡建议启用以节省显存
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
    # 打开fp16精度，V100、P100、T4等显卡建议启用以节省显存
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
    # 使用CPU进行推理，需要约32GB内存
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="cpu", trust_remote_code=True).eval()
    # 默认使用自动模式，根据设备自动选择精度
    model = AutoModelForCausalLM.from_pretrained(qwen_path, device_map="auto", trust_remote_code=True).eval()

    # 可指定不同的生成长度、top_p等相关超参
    model.generation_config = GenerationConfig.from_pretrained(qwen_path, trust_remote_code=True)

    return model, tokenizer

def chat_test(model, tokenizer):
    # 第一轮对话 1st dialogue turn
    response, history = model.chat(tokenizer, "你好", history=None)
    print(response)
    # 你好！很高兴为你提供帮助。

    # 第二轮对话 2nd dialogue turn
    response, history = model.chat(tokenizer, "给我讲一个年轻人奋斗创业最终取得成功的故事。", history=history)
    print(response)

    # 第三轮对话 3rd dialogue turn
    response, history = model.chat(tokenizer, "给这个故事起一个标题", history=history)
    print(response)

    # Qwen-1.8B-Chat现在可以通过调整系统指令（System Prompt），实现角色扮演，语言风格迁移，任务设定，行为设定等能力。
    response, _ = model.chat(tokenizer, "你好呀", history=None, system="请用二次元可爱语气和我说话")
    print(response)

    response, _ = model.chat(tokenizer, "我同事工作认真", history=None,
                             system="你会根据需要写漂亮的赞美")
    print(response)

def load_model_Quantization():
    from modelscope import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    tokenizer = AutoTokenizer.from_pretrained(qwen_path, revision='master', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        qwen_path,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    return model

def load_model_modelscope_qwen2():
    from modelscope import AutoModelForCausalLM, AutoTokenizer
    device = "cuda"  # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained(
        # "qwen/Qwen2-1.5B-Instruct",
        qwen_path,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(qwen_path)

    prompt = "给我简单介绍下大语言模型."
    messages = [
        {"role": "system", "content": "你是个很好的助手."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    pass

def load_model_transformers_qwen2():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = qwen_path
    device = "cuda"  # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained(
        qwen_path,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = "你是谁?"
    messages = [
        # {"role": "system", "content": "你是个很好的助手."},
        {"role": "system", "content": "你是个有用的助手."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=500,
        eos_token_id=151643,
        # pad_token_id=151643
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    pass

if __name__ == '__main__':
    qwen_path = r"C:\Users\11658\.cache\modelscope\hub\qwen\Qwen-1_8B-Chat-Int4"
    qwen_path = r"qwen/Qwen2-7B-Instruct" # "qwen/Qwen2-1.5B-Instruct"  "Qwen/Qwen2-7B-Instruct"
    qwen_path = r"D:\03GitHub\Qwen2\examples\sft\output" # "qwen/Qwen2-1.5B-Instruct"  "Qwen/Qwen2-7B-Instruct"
    # qwen_path = r"C:\Users\11658\.cache\modelscope\hub\Qwen\Qwen2-0___5B" # "qwen/Qwen2-1.5B-Instruct"  "Qwen/Qwen2-7B-Instruct"

    # model, tokenizer = load_model_transformers()
    # model = load_model_Quantization()
    # chat_test(model, tokenizer)

    load_model_transformers_qwen2()

    pass























