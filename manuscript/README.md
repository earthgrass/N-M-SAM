# APBR-SAM：无阈值 Geometry-first 论文修订

编译入口：`main.tex`。单栏Times；英文正文分为 `sections/` 子文件，Abstract、Introduction、Contributions、Method/Algorithm可独立编辑。

新增方法定义：APBR首删 → 虚拟OBS条件化 → 纯条件OBS次删 → 一次物理联合补偿。

- `TAU_ORIGIN_AUDIT.md`：逐项回答旧τ来源、授权、实现、结果及Test前冻结。
- `ALGORITHM_VERSION_MAPPING.md`：物理条件版、近优几何版、本文无τ版。
- `RESULT_PROVENANCE_CHECK.md`：结果来源及新旧隔离。
- `NARRATIVE_REVISION_REPORT.md`：写作和数学对象对应。
- `PENDING_EXPERIMENTS.md`：仅记录待验证项，未执行实验。
- `before/`：原稿与历史表，不被新main引入。
- `revision.diff`：相对于基准提交的文字差异。

独立Git分支：`research/apbr-first-conditional-obs-paper`。
基准提交：`e0527ce1730c0ccacbd9625fd3758cb902b72ddd`。
原工程仓库、原稿、checkpoint和实验代码未修改。

构建顺序：XeLaTeX → BibTeX → XeLaTeX → XeLaTeX。
