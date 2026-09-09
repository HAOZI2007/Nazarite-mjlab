# Nazarite 版本提交规范

本规范用于区分可复现源码、可审核数据和本地生成产物。提交前以仓库根目录
`.gitignore` 为强制规则。

## 可以提交

- `Train/Nazarite/Nazarite-src/` 下的 Python 源码与任务配置。
- `Train/Nazarite/tools/` 下的离线工具与工具说明，但不包括本地
  `tools/smp_dataset/`。
- `Train/Nazarite/tests/` 下的自动化测试。
- `docs/`、`README.md`、`pyproject.toml`、`uv.lock` 等文档和环境定义。
- 机器人 MJCF、地形配置以及运行代码所需的小型静态资源。

## 不得提交

- 任意 `output/` 目录及其中的预处理结果、验证结果、训练中间文件和临时图像。
- `logs/`、`wandb/`、视频、ONNX、benchmark 与本地调试输出。
- `.venv/`、缓存目录、`__pycache__/` 和编辑器本地状态。
- `*.pt`、`*.pth`、`*.onnx` 等模型权重或导出文件；确需共享时应单独使用
  release、对象存储或经项目确认的 Git LFS 方案。
- 完整外部数据集以及 `Train/Nazarite/tools/smp_dataset/` 下的所有原始数据、
  派生数据、统计文件、manifest、说明文件和扩散 prior。

## SMP 数据约定

SMP 数据和扩散模型统一放在本地
`Train/Nazarite/tools/smp_dataset/`，该目录整体禁止提交。数据生成过程仍输出到
同样不可提交的 `output/`，验收后再显式复制所需子集到本地数据集目录。

## 提交前检查

```bash
git status --short
git status --ignored --short
```

确认 `output/`、训练日志和模型权重只出现在 ignored 列表中，再运行与改动相关的
Ruff、Pyright 和 Pytest 检查。
