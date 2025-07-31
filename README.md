HGCN with KG Enhancement
This project is based on the HGCN (Hierarchical Graph Convolutional Network) model, enhanced with Knowledge Graph (KG) to improve text representation. The main workflow includes: extracting entities using BERT-NER, obtaining entity vectors with Wikipedia2Vec, fusing KG vectors into training data, and modifying the HGCN model to support KG enhancement.
Environment Setup
Python 3.6+
PyTorch 1.7+
transformers
wikipedia2vec
Other dependencies are listed in requirements.txt
Model Download
BERT-NER Model
Download address: checkpoints/entity_enhanced/best_model.pt
Description: NER model based on bert-base-uncased for entity extraction.
Wikipedia2Vec Model
Download address: models/wiki2vec/enwiki_20180420_300d.pkl
Description: Pre-trained Wikipedia2Vec model for obtaining entity vectors.
SciBERT Model
Download address: allenai/scibert_scivocab_uncased (automatically downloaded via HuggingFace)
Description: Used for text encoding and classification.
Notes
Ensure that model files have been downloaded to the specified paths.
Data files must be in JSON or JSONL format, with each data entry containing fields such as title, abstract, label, etc.
If using demo data, replace filenames in commands with sample_*.json.
Entity-Enhanced HGCN
This project is an extension of the original HGCN, enhancing text representation by integrating entity information and knowledge graph embeddings. We have implemented the following features:
Datasets and Pre-trained Models
2. Download Pre-trained Models
Wikipedia2Vec Model
bash
# Create Wiki2Vec directory
mkdir -p models/wiki2vec

# Download Wiki2Vec pre-trained model
python src/download_wiki2vec.py --output_dir models/wiki2vec
3. Verify Downloads
bash
# Check dataset files
ls data/exaapd/
# Should see: train.json, dev.json, test.json, etc.

# Check model files
ls models/bert/
ls models/wiki2vec/
# Should see corresponding model files


Environment Requirements
bash
Python 3.7+
PyTorch 1.8+
transformers
numpy
tqdm
Wikipedia2Vec
scikit-learn

Notes
Ensure sufficient disk space for storing the Wikipedia2Vec model
It is recommended to use a GPU for training
Adjust batch_size according to the actual GPU memory size
The model size can be controlled by adjusting hidden_dim
The first run requires downloading pre-trained models, which may take a long time
HGCN: 层次图卷积网络用于结构化长文档分类
本项目实现了层次图卷积网络（HGCN）用于结构化长文档分类，并提供了从原始数据处理到模型训练的完整流程。项目特别关注增强嵌入（enhanced embeddings）的生成和使用，以提升模型性能。
完整项目流程
1. 环境准备
bash
# 创建并激活conda环境
conda create -n hgcn python=3.8
conda activate hgcn

# 安装依赖包
pip install torch==1.10.0 transformers==2.1.1 numpy scipy tqdm pandas wikipedia2vec
2. 数据处理流程
2.1 命名实体识别（NER）
提取文本中的实体信息：

bash
python ner.py data/exAAPD_train.json 0

输出文件：ner.csv
2.2 下载词向量模型
bash
python download_wikipedia2vec.py

下载模型：wikipedia2vec_models/wiki.en.vec
2.3 提取实体向量
bash
python extract_vector.py

输出文件：entity_embeddings.csv
2.4 融合向量到原始数据
bash
python Fusion_vector.py

输出文件：

merged_kg_dataset.json
enhanced_embeddings.json（作为 HGCN 训练的基础数据）
2.5 生成标签
bash
python generate_labels.py

输出文件：enhanced_embeddings_with_labels.json
3. 模型训练
3.1 配置训练参数
可在 run_enhanced_embeddings_training.sh 中修改以下参数：

CUDA_VISIBLE_DEVICES：指定 GPU 设备
DATA_DIR：数据文件路径
SAVE_PATH：模型保存路径
其他训练参数（批大小、学习率等）
3.2 运行训练脚本
bash
chmod +x run_enhanced_embeddings_training.sh
./run_enhanced_embeddings_training.sh
项目结构
plaintext
HGCN-master/
├── src/
│   ├── train_enhanced_embeddings.py  # 增强嵌入训练脚本
│   ├── train.py                      # 普通训练脚本
│   └── args.py                       # 参数配置
├── common/                           # 公共模块
│   ├── evaluators/                   # 评估器
│   └── trainers/                     # 训练器
├── datasets/                         # 数据集处理
│   ├── bert_processors/              # BERT处理器
│   └── bow_processors/               # BoW处理器
├── utils/                            # 工具函数
├── ner.py                            # NER处理脚本
├── download_wikipedia2vec.py         # 下载词向量模型
├── extract_vector.py                 # 提取实体向量
├── Fusion_vector.py                  # 融合向量
├── generate_labels.py                # 生成标签
├── run_enhanced_embeddings_training.sh  # 训练运行脚本
└── model_checkpoints/                # 模型保存目录
数据格式说明
输入数据格式
enhanced_embeddings.json 包含以下字段：

id：唯一标识符
title：文档标题
kg_embeddings：知识图谱嵌入向量
kg_entities：相关实体列表
enhanced_vector：增强向量
vector_dim：向量维度
输出数据格式
训练好的模型保存在 ./model_checkpoints/enhanced_embeddings/
包含模型权重、配置文件和训练日志
注意事项
确保在运行训练脚本前完成所有数据处理步骤
根据您的硬件配置调整批大小和序列长度
训练过程中需要 CUDA 支持（推荐使用 GPU）
如需修改模型架构，请查看 src/train_enhanced_embeddings.py
故障排除
CUDA 内存不足：尝试减小批大小或序列长度
模型收敛不佳：调整学习率或增加训练轮数
数据格式错误：检查输入 JSON 文件是否符合要求
