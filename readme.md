# PRPS（案件侧写安全评估）

本仓库包含一套科研脚本，用于在“案件侧写（profiling）”场景下评估大模型输出是否产生“可操作的违法/犯罪指导（instruction）”风险：  
- **Target 模型**：火山引擎 Ark（豆包）对案件证据链进行推演式回答  
- **Judge 模型**：通过 Polo API（OpenAI 兼容接口）对回答进行打分与风险分级  
- **分析脚本**：对评估结果进行统计可视化（条形图/热力图/直方图/箱线图等）

## 目录结构

- `Code/Attack_Eval_M1.py`：主评估脚本（Target 生成 + Judge 评分 + 输出结果 CSV）
- `Code/Result_Analysis.py`：对结果 Excel 进行图表分析并保存图片
- `Code/process_safebench.py`：将 SafeBench 指令转换为中文“数据集生成 prompt”（需要自行修改输入/输出路径）
- `Code/quchong.py`：CSV 去重脚本（需要自行修改输入/输出路径）
- `CrimeTrace-500.csv` / `CrimeTrace-500.xlsx`：CrimeTrace 数据集（500 条，含 3 条证据字段）
- `CrimeTrace-500-Result.xlsx`：示例/已有评估结果（用于 `Result_Analysis.py`）
- `Charts/`：仓库内自带的示例图表图片（注意：分析脚本默认输出目录为 `Chart/`，会自动创建）
- `提示词&模型参数.txt`：核心 prompt 与参数记录
- `测试日志&图表说明.txt`：实验日志、判分逻辑讨论与图表解读

## 环境准备

建议使用 **Python 3.10+** 的干净虚拟环境（避免 Conda 环境中 NumPy/PyArrow 等二进制依赖版本不一致导致的导入报错）。

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

## 关键配置（API Key）

`Code/Attack_Eval_M1.py` 通过环境变量读取密钥：

- `VOLC_ARK_API_KEY`：火山引擎 Ark（豆包）API Key
- `POLO_API_KEY`：Polo API Key（OpenAI 兼容 Chat Completions）

PowerShell 示例：

```powershell
$env:VOLC_ARK_API_KEY="..."
$env:POLO_API_KEY="..."
```

## 运行评估（Target vs Judge）

1) 打开 `Code/Attack_Eval_M1.py`，确认/修改以下配置（文件开头的“配置区域”）：
- `INPUT_FILE`：输入数据 CSV 路径（仓库自带 `CrimeTrace-500.csv`，你也可以换成子集文件）
- `TARGET_MODEL_ID`：豆包模型 ID（示例：`doubao-1-5-pro-32k-250115`）
- `JUDGE_MODEL_NAME`：裁判模型名（示例：`gpt-5.1`）
- `POLO_API_URL`：Judge 接口地址（默认 `https://poloai.top/v1/chat/completions`）

2) 运行脚本：

```bash
python Code/Attack_Eval_M1.py
```

3) 输出：脚本会生成一个结果 CSV（文件名包含 Target/Judge 标识），核心字段包括：
- `序号`、`案件类型`
- `Target_Response`
- `Judge_Analysis`
- `Score_Instruction` / `Score_Entity` / `Score_AntiForensics`
- `Risk_Level`（`High_Critical` / `Medium_Risky` / `Low_Safe`）
- `Is_Jailbroken`

## 运行：结果分析与制图

`Code/Result_Analysis.py` 默认读取根目录的 `CrimeTrace-500-Result.xlsx`（`Sheet1`），并在运行目录下创建 `Chart/` 用于保存图表：

```bash
python Code/Result_Analysis.py
```

提示：
- 脚本在 Windows 下默认设置字体为 `SimHei` 以显示中文；若系统缺少该字体，图中中文可能显示为方块。
- 运行过程中会弹出 matplotlib 窗口（`plt.show()`）；如在无 GUI 环境运行，可自行改为仅保存图片（本仓库不做代码改动）。

## SafeBench 数据处理（可选）

仓库内提供了两个辅助脚本用于生成/清洗数据，但都包含**硬编码的本地路径**，需要你先改成自己的路径后再运行：

- `Code/process_safebench.py`：读取 `safebench.csv`（示例路径为桌面），生成 `dataset_prompt` 列并输出新 CSV
- `Code/quchong.py`：对处理后的 CSV 去重并保存

## 实验设定与判分逻辑

具体的 prompt、温度参数、以及“如何区分 profiling 与 instruction”的判分讨论见：
- `提示词&模型参数.txt`
- `测试日志&图表说明.txt`

其中记录了：
- attack prompt：要求模型基于证据链从“物资/步骤/行为/目标”四维推演
- judge prompt：三维度打分（实操性/实体敏感度/反侦查技术深度）及风险分级阈值设定

