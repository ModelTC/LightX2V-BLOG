# My Blog

基于 Jekyll 构建的个人博客，托管在 GitHub Pages，完全免费。

## 本地运行

```bash
# 安装依赖
bundle install

# 启动开发服务器（支持热更新）
bundle exec jekyll serve --livereload

# 浏览器访问
open http://localhost:4000
```

## 写新文章

在 `_posts/` 目录下创建 Markdown 文件，文件名格式：

```
YYYY-MM-DD-your-post-title.md
```

文章头部必须包含 Front Matter：

```markdown
---
layout: post
title: "文章标题"
date: 2026-04-14
tags: [标签1, 标签2]
---

文章正文从这里开始...
```

## 部署

推送到 `main` 分支，GitHub Actions 会自动构建并部署到 GitHub Pages。

## 自定义

- 修改 `_config.yml` 中的博客标题、描述、作者信息
- 修改 `assets/css/style.css` 调整样式
- 修改 `about.md` 填写关于页面
