# UniGuard reproduction

本项目是："UniGuard: Safety Guardrails for Multimodal Large Language Models against Jailbreak Attacks" 的复现

原项目：https://anonymous.4open.science/r/UniGuard/README.md

UniGuard 是一个面向多模态输入的安全护栏框架，旨在最小化 LLaVA‑v1.5 等多模态大模型生成有害回复的概率。作者还验证了该护栏能迁移到其他 MLLM（如 GPT‑4V、MiniGPT‑4、InstructBLIP），扩大了方法的适用性。

---

## 项目结构 🔧

- **`get_ppl.py`：** 计算样本的困惑度指标
- **`get_metric.py`：** 使用 Perspective API 计算指标的脚本
- **`cal_metrics.py`:** 汇总Perspective API 指标
- **`eval_configs`：** 模型评估配置（包含 llama 与 MiniGPT‑4）
- **`image_safety_patch.py`, `text_safety_patch.py`：** 生成图像/文本安全补丁的脚本
- **`smooth.py`：** smoothLLM 方法的实现
- **`instructblip_*.py`：** 与 InstructBLIP 相关的防御、受约束/非受约束攻击和问答脚本
- **`lavis`：** InstructBLIP 子模块（数据集、模型、处理器、runner、task 等）
- **`metric`：** Detoxify 与 Perspective API 的实现
- **`minigpt_*.py`：** 与 MiniGPT‑4 相关的推理、攻击与问答脚本
- **`requirements.txt`：** 依赖包清单
- **`scripts`：** 用于运行所有实验的 shell 脚本
- **`utils.py`：** 常用工具函数（图片加载/预处理等）
- **`visual`：** 用于可视化 InstructBLIP 与 MiniGPT‑4 毒性结果的脚本
- **`text_patch_heuristic`：** 预定义文本护栏
- **`text_patch_optimized`：** 优化后的文本护栏

---

## 快速开始（Setup） 📥

1. 克隆仓库：
   ```bash
   git clone https://github.com/Yang-Shihui/UniGuard-reproduction.git
   cd UniGuard-reproduction
   ```

2. 安装依赖（需要 Python 3.10+）：
   ```bash
   pip install -r requirements.txt
   ```

3. 数据集准备：
   从 [Google Drive](https://drive.google.com/drive/folders/14vdgC4L-Je6egzmVOfVczQ3-j-IzBQio?usp=sharing) 下载两个文件并放到项目目录，然后解压：
   ```bash
   tar -xzvf adversarial_qna_images.tar.gz
   tar -xzvf unconstrained_attack_images.tar.gz
   ```

> 注意：部分评估（例如 A‑OKVQA / MM‑Vet）需要额外下载并适配数据集格式，仓库未包含这些数据的原始包。

---

