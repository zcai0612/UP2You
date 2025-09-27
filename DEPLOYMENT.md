# 静态 HTML 部署到 GitHub Pages

## 部署步骤

### 1. 准备工作
确保你的仓库包含以下文件：
- `index.html` - 主页面
- `public/carousel.js` - 轮播功能 JavaScript
- `src/assets/` - 所有媒体资源
- `.nojekyll` - 禁用 Jekyll 处理
- `.github/workflows/deploy.yml` - GitHub Actions 工作流

### 2. GitHub Pages 设置

1. 进入你的 GitHub 仓库
2. 点击 **Settings** 标签
3. 在左侧菜单中找到 **Pages**
4. 在 **Source** 中选择 **GitHub Actions**
5. 保存设置

### 3. 自动部署

每次推送到 `main` 或 `master` 分支时，GitHub Actions 会自动：
1. 检出代码
2. 配置 GitHub Pages
3. 上传整个项目作为静态网站
4. 部署到 GitHub Pages

### 4. 访问网站

部署完成后，你的网站将在以下地址可用：
```
https://yourusername.github.io/repository-name/
```

### 5. 故障排除

#### 常见问题：

1. **Jekyll 错误**: 确保根目录有 `.nojekyll` 文件
2. **资源加载失败**: 检查 `index.html` 和 `carousel.js` 中的路径是否正确
3. **视频不播放**: 确保视频文件在 `src/assets/` 目录中，且路径正确

#### 检查部署状态：
1. 在仓库中点击 **Actions** 标签
2. 查看最新的工作流运行状态
3. 如有错误，点击失败的工作流查看详细日志

### 6. 本地测试

你可以使用以下任意方式在本地测试：

```bash
# 使用 Python 3
python -m http.server 8000

# 使用 Python 2
python -m SimpleHTTPServer 8000

# 使用 Node.js
npx serve .
```

然后在浏览器中访问 `http://localhost:8000`

### 7. 项目结构

```
website_source/
├── index.html              # 主页面
├── .nojekyll              # 禁用 Jekyll
├── public/
│   └── carousel.js        # 轮播功能
├── src/
│   └── assets/           # 媒体资源
│       ├── *.mp4         # 视频文件
│       ├── *.png         # 图片文件
│       └── corr_map/     # 相关性地图视频
└── .github/
    └── workflows/
        └── deploy.yml    # 部署工作流
```

## 注意事项

1. 确保所有资源文件都已提交到 Git 仓库
2. 大文件可能会影响部署时间和网站加载速度
3. GitHub Pages 有 1GB 的空间限制
4. 建议压缩大型媒体文件以提高加载速度
