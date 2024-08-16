# 模型下载 Qwen-1_8B  Qwen-1_8B-Chat,
# Qwen-1.8B, Qwen-7B, Qwen-14B, and Qwen-72B
from modelscope import snapshot_download
# model_dir = snapshot_download('qwen/Qwen-1_8B-Chat')

# Qwen2  Qwen2-7B-Instruct
# Qwen2-0.5B, Qwen2-1.5B, Qwen2-7B, Qwen2-57B-A14B, and Qwen2-72B;
model_dir = snapshot_download('Qwen/Qwen2-0.5B')


#数据集下载
# from modelscope.msdatasets import MsDataset
# ds =  MsDataset.load('BJQW14B/bs_challenge_financial_14b_dataset', subset_name='pdf_txt_file', split='train')
#您可按需配置 subset_name、split，参照“快速使用”示例代码  sqlite_db pdf_file pdf_txt_file




