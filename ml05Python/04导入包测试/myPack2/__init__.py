import sys, os
from pathlib import Path
# algseg添加路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
sys.path.insert(0,str(FILE.parents[1]))
sys.path.insert(0,str(ROOT))
print("添加路径完成：", ROOT)
print("\n",sys.path)
