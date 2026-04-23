# LightX2V Blog

LightX2V Repository: https://github.com/ModelTC/LightX2V

Blog URL: http://light-ai.top/LightX2V-BLOG/

## How to Add a New Blog Post

Create a markdown file in the `_articles` folder with your content.

## ⚠️ Blog Format Requirements

### Front Matter (Required)

Every article must include YAML front matter at the beginning with the following format:

```yaml
---
layout: post
title: "Article Title"
author: "Author Name"
date: YYYY-MM-DD
tags: [tag1, tag2, ...]
---
```

**Field Descriptions:**

| Field | Required | Description |
|-------|----------|-------------|
| layout | ✓ | Must be `post` |
| title | ✓ | Article title, wrapped in quotes |
| author | ✓ | Author name |
| date | ✓ | Publication date in format: YYYY-MM-DD |
| tags | ✓ | List of tags separated by commas in brackets |

## Deployment

The project uses GitHub Pages for automatic deployment. After committing to the main branch, the content will be automatically built and published to: http://light-ai.top/LightX2V-BLOG/
