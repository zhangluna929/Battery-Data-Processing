# Battery Data Processing and Electrochemical Analysis / 电池数据处理与电化学分析

作者 / Author: **lunazhang**

---

## 1  Project Overview / 项目概述

This toolkit is conceived as an end-to-end computational framework for the rigorous interrogation of electrochemical cycling data, spanning raw acquisition pipelines to high-level degradation prognosis.  The codebase encapsulates data normalization layers, physics-informed analytics, surrogate modeling and interactive visualization, thereby enabling quantitative insight into kinetic, thermodynamic and ageing phenomena of advanced lithium-ion and solid-state chemistries.

该工具链旨在提供一条端到端的计算框架，用于对电化学循环数据进行严格剖析，覆盖原始数据采集、物理机理分析、替代模型构建以及交互可视化的全过程。代码库内嵌数据规范化层、基于机理的分析算法、代理模型以及可交互图形界面，为研究者定量揭示先进锂离子及固态体系的动力学、热力学及老化机制。

---

## 2  Core Capabilities / 核心能力

1. Unified loader system capable of parsing Arbin, NEWARE, Bio-Logic and generic CSV/Excel formats.  
   统一的数据加载系统，可解析 Arbin、NEWARE、Bio-Logic 以及通用 CSV/Excel 格式。
2. Robust preprocessing: missing-data imputation, Z-score outlier rejection, configurable low-pass Butterworth smoothing.  
   稳健的预处理流程：缺失值线性插值、Z-score 异常点剔除、可配置 Butterworth 低通滤波。
3. Instrument calibration via multivariate linear regression against temperature drift.  
   基于温度漂移的多元线性回归设备校准。
4. High-precision capacity integration (trapezoidal rule, sub-second resolution).  
   高精度容量积分（梯形法、亚秒级时间分辨率）。
5. Incremental Capacity Analysis (ICA, dQ/dV) with Savitzky–Golay denoising and voltage grid interpolation.  
   增量容量分析（ICA, dQ/dV）：Savitzky–Golay 去噪与电压网格插值。
6. Differential voltage & Coulombic efficiency per cycle; automatic half-cycle segmentation.  
   周期级差分电压与库仑效率，自动半周期分段。
7. DC internal resistance (DCIR) event detection (ΔV/ΔI) with threshold gating.  
   直流内阻（DCIR）事件检测（ΔV/ΔI）及阈值筛选。
8. Feature engineering: dI/dt, mean DCIR, engineered SOC.  
   特征工程：电流变化率 dI/dt、平均 DCIR、工程化 SOC。
9. Machine-learning pipeline (Scikit-learn Pipeline + Random Forest) with YAML-declared feature/target sets.  
   机器学习管线（Scikit-learn Pipeline + 随机森林），特征/目标由 YAML 声明。
10. Coulomb-counting SOC estimator with OCV correction, ready for EKF extension.  
    带 OCV 校正的库仑计量 SOC 估计器，预留 EKF 升级接口。
11. Early-life capacity trend based RUL predictor (linear regression).  
    基于早期容量衰减趋势的 RUL 预测器（线性回归）。
12. Streamlit + Plotly interactive dashboard for ICA curves, capacity fade and DCIR scatter.  
    Streamlit + Plotly 交互式仪表盘，可视化 ICA 曲线、容量衰减及 DCIR 散点。
13. Fully YAML-parameterized; zero magic numbers in source code.  
    全 YAML 参数化，源码中无硬编码常数。

---

## 3  Directory Layout / 目录结构

```
battery_analyzer/                 # Core package 
├── loaders/                      # Data loaders
├── processing/                   # Cleaning & calibration
├── analysis/                     # Electrochemical analytics
├── models/                       # Machine-learning & estimators
├── utils/                        # Utilities
configs/                          # YAML configuration files
README.md                         # Project documentation
requirements.txt                  # Python dependencies
main.py                           # CLI entry point
dashboard.py                      # Streamlit dashboard
```

---

## 4  Quick Start / 快速开始

```bash
pip install -r requirements.txt
python main.py --input-dir raw_data --output-dir processed_data --train-model
streamlit run dashboard.py
```

---

## 5  Configuration Schema / 配置架构

All numerical thresholds, column mappings, filter orders, ML feature sets and SOC/OCV parameters are declared in `configs/default_config.yml`. Researchers can maintain multiple scenario-specific YAMLs without touching code.

所有数值阈值、列名映射、滤波器参数、机器学习特征以及 SOC/OCV 参数均在 `configs/default_config.yml` 中声明。科研人员可维护多套场景化 YAML，而无需修改源代码。

---

## 6  Extensibility & Roadmap / 可扩展性与未来规划

* Plug-in loader interfaces for emerging solid-state cyclers.  
  新型固态电池测试仪数据加载插件接口。
* EKF/UKF-based SOC algorithm, parameter identification via HPPC.  
  基于 EKF/UKF 的 SOC 算法，结合 HPPC 数据进行参数辨识。
* GPU-accelerated gradient boosting (LightGBM / XGBoost) for SOH & RUL.  
  引入 GPU 加速的梯度提升模型（LightGBM / XGBoost）用于 SOH 与 RUL。
* Automated LaTeX/PDF report generator for regulatory submissions.  
  自动化报告生成（LaTeX/PDF）以支持法规认证提交。

---

## 7  Citation / 引用

If this toolbox accelerates your research, please cite:  
> lunazhang, "Battery Data Processing and Electrochemical Analysis Toolkit", 2025.

若本工具对您的研究有所助益，请引用：  
> lunazhang，《电池数据处理与电化学分析工具集》，2025。

---

## 8  License / 许可证

MIT
