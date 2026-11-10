# Build check

编译器：本机MiKTeX XeLaTeX，单栏article/Times New Roman。

构建：XeLaTeX + BibTeX + XeLaTeX多遍，最终记录由record_build生成。

初次record_build遇到Windows输出编码和辅助检查器的嵌套input解析问题；论文本身成功编译。已将Contributions的input提升到main直接引用，并在工具调用时启用Python UTF-8。未改实验环境。

最终结果：

- 8页，单栏，Times New Roman；Algorithm1全部在第4页。
- Abstract为168英文词；三条贡献的引导句与列表保持同页。
- 无LaTeX错误、未定义引用、未定义交叉引用、overfull或缺失文件。
- 保留4项underfull排版提示；逐页图像检查没有裁切、重叠或不可读区域。
- Citation key check PASS；draft research quality gate PASS。
- PDF文本中无tau/τ/near-optimal，也没有旧77.63或77.43。
- 数学同行检查：full-active-set R、两步选择、一次joint、固定D约束解及成本分解PASS。
- 来源文字复核PASS；将尚未执行的评测明确写为pending/future。
- 原main.tex、原PDF及两份原表SHA256与本轮开始时相同。

当前paper_state保持drafting：写作交付完成，实验仍为PENDING；未标为submission-ready。
