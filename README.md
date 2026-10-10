# 知识库

基于 [MkDocs](https://www.mkdocs.org/) + [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) 构建的静态知识库。

## 本地预览

```bash
pip install -r requirements.txt
mkdocs serve          # 打开 http://127.0.0.1:8000
```

## 构建静态站点

```bash
mkdocs build          # 产物在 site/
```

## 新增内容

导航在 `mkdocs.yml` 的 `nav:` 中**显式声明**（目录名保持英文，导航标签用中文）：

1. 建目录，用 `序号-英文短名` 命名（如 `1-architecture/`）；
2. 目录内写 `index.md`；
3. 在 `mkdocs.yml` 的 `nav:` 里加一条分组，把该目录的 `index.md` 作为**第一个子项**（不写标题）：

   ```yaml
   - 架构与骨干:
       - 1-vla/1-architecture/index.md
       - 双专家共享注意力: 1-vla/1-architecture/1-dual-expert/index.md
   ```

4. 回到上一层 `index.md`，在表格里补一行链接。

> 分组标题即该分组的 `index.md` 页面（依赖 `navigation.indexes` 特性），
> 所以标题可点击进入 Summary，且不会再重复渲染出一条同名子项。

页面之间用**相对路径链接**互跳，例如 `[VLA 模块](1-vla/index.md)`。

## 发布到 GitHub Pages

推送到 `main` 分支后，`.github/workflows/deploy.yml` 会自动执行 `mkdocs gh-deploy`。
首次使用需在仓库 Settings → Pages 中把 Source 设为 `gh-pages` 分支。

## 目录结构

```
KnowledgeBase/
├── mkdocs.yml                    # 站点配置（导航在 nav: 中显式声明）
├── requirements.txt              # Python 依赖
├── docs/
│   ├── index.md                  # 首页：知识库说明
│   ├── 1-vla/                    # 模块 1：VLA
│   │   ├── index.md              #   模块 Summary
│   │   ├── 1-architecture/       #   子方向：架构与骨干
│   │   │   ├── index.md          #     子方向 Summary
│   │   │   ├── 1-dual-expert/    #       细分方向
│   │   │   │   └── index.md
│   │   │   └── papers/           #       本方向论文源文件
│   │   │       ├── index.md
│   │   │       └── pi0_model_architecture.md
│   │   └── 2-action-generation/  #   子方向：动作生成范式
│   │       ├── index.md
│   │       ├── 1-flow-matching/
│   │       │   └── index.md
│   │       └── papers/
│   ├── 2-world-model/            # 模块 2：世界模型（结构同上）
│   └── 3-rl/                     # 模块 3：强化学习（结构同上）
└── .github/workflows/deploy.yml  # 自动部署
```

### 组织约定

- **模块目录**：`index.md`（模块 Summary）+ 若干子方向目录
- **子方向目录**：`index.md`（子方向 Summary）+ 若干细分方向目录 + `papers/`
- **细分方向目录**：`index.md`
