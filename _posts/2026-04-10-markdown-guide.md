---
layout: post
title: "Markdown 写作指南"
date: 2026-04-10
tags: [Markdown, 写作]
---

这篇文章演示 Markdown 的各种用法，同时也是样式测试页面。你可以删掉这篇文章，或者保留它作为参考。

## 标题层级

# H1 标题（通常不在文章内使用，已在文章头部）
## H2 标题
### H3 标题
#### H4 标题

---

## 文字格式

普通段落文字。这是正文文字的样子，使用衬线字体增强阅读体验。

**加粗文字** 和 *斜体文字* 以及 ~~删除线~~。

行内代码：`const greeting = "Hello, World!"`

---

## 引用

> 好的文章不是写出来的，而是改出来的。
>
> —— 无名氏

---

## 列表

**无序列表：**

- 第一项
- 第二项
  - 嵌套子项
  - 另一个子项
- 第三项

**有序列表：**

1. 确定主题
2. 写初稿，不要停
3. 休息一下
4. 修改打磨
5. 发布

---

## 代码块

```javascript
// 一个简单的函数
function greet(name) {
  return `Hello, ${name}!`;
}

console.log(greet('World'));
```

```python
# Python 示例
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print([fibonacci(i) for i in range(10)])
```

---

## 表格

| 语言       | 创建年份 | 主要用途          |
|-----------|---------|-----------------|
| Python    | 1991    | 数据科学、后端    |
| JavaScript| 1995    | 前端、全栈       |
| Go        | 2009    | 后端、系统工具    |
| Rust      | 2010    | 系统编程、WebAssembly |

---

## 链接和图片

[访问 Jekyll 官网](https://jekyllrb.com)

外部链接默认在当前窗口打开，你可以在 Markdown 中加上 HTML 属性：

<a href="https://github.com" target="_blank" rel="noopener">在新窗口打开 GitHub</a>

---

这篇文章展示了主要的 Markdown 元素。开始写你自己的文章吧！
