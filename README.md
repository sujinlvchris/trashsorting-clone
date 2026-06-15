# BinGo 泰国垃圾分类识别系统

这是一个部署在 Vercel 上的泰国垃圾分类 Web 应用。用户可以上传垃圾图片，也可以打开摄像头实时扫描，系统会根据泰国/曼谷常见垃圾分类方式给出推荐投放类别、垃圾桶颜色、置信度和处理建议。

公网地址：

```text
https://bingo-sorter.vercel.app/
```

## 主要功能

- 上传图片分类：支持 JPG、PNG、WebP。
- 实时摄像头分类：支持 Start camera、Stop camera、Switch camera、Classify now。
- 每 1.5 秒从摄像头视频抽一帧到 canvas，再发送到后端分类。
- 连续两次同类结果才自动更新，减少画面抖动造成的结果跳变。
- 支持英文、泰文、中文分类名称。
- 返回垃圾桶颜色、分类依据、泰国投放建议、避免事项和置信度。
- 后端接口部署在 Vercel Serverless Function：`POST /api/classify`。
- 默认后端调用 Hugging Face Space 中的不量化 ViT 模型进行图片分类。
- 也可以通过环境变量切回 OpenAI Responses API。

## 分类规则

当前版本按泰国/曼谷常见四分类实现：

| 颜色 | 英文分类 | 泰文 | 中文 |
| --- | --- | --- | --- |
| Green | Food / Organic waste | ขยะเศษอาหาร / ขยะอินทรีย์ | 厨余 / 有机垃圾 |
| Yellow | Recyclable waste | ขยะรีไซเคิล | 可回收物 |
| Blue | General waste | ขยะทั่วไป | 一般垃圾 |
| Orange / Red | Hazardous waste | ขยะอันตราย | 有害垃圾 |

参考方向来自 Greener Bangkok 的社区垃圾分类指南：

```text
https://greener.bangkok.go.th/en/waste-recycle/community-waste-guide/
```

## 项目结构

```text
.
├── index.html
├── styles.css
├── app.js
├── assets/
│   ├── bin-blue.png
│   ├── bin-green.png
│   ├── bin-red.png
│   ├── bin-yellow.png
│   └── bins-logo.png
└── api/
    ├── classify.js
    └── lib/
        ├── huggingFaceWasteClassifier.js
        ├── imageInput.js
        ├── openaiWasteClassifier.js
        └── thaiWasteRules.js
```

## 后端接口

接口地址：

```http
POST /api/classify
```

请求示例：

```json
{
  "image": "data:image/png;base64,...",
  "mimeType": "image/png",
  "fileName": "camera-frame.png",
  "source": "camera"
}
```

返回示例：

```json
{
  "ok": true,
  "bin": "green",
  "label": "Green bin · Food / Organic waste",
  "categoryDetail": "Food waste and organic waste",
  "thaiName": "ขยะเศษอาหาร / ขยะอินทรีย์",
  "chineseName": "厨余 / 有机垃圾",
  "confidence": 82,
  "needsManualCheck": false
}
```

## Hugging Face Space API 说明

当前默认后端流程：

1. Vercel `/api/classify` 接收上传图片或摄像头帧。
2. Vercel 把图片转发给 Hugging Face Space：

```text
https://chrissujinlv-bingo-thai-waste-api.hf.space/classify
```

3. Space 加载 Hugging Face Model Repo 中的不量化 ViT 模型：

```text
ChrisSujinlv/bingo-thai-four-bin-waste-vit
```

4. Space 返回 `green/yellow/blue/red` 和置信度。
5. Vercel 把结果映射为泰国四桶分类建议并返回给前端。

Vercel 环境变量：

```text
WASTE_CLASSIFIER_PROVIDER=huggingface-space
HF_SPACE_API_URL=https://chrissujinlv-bingo-thai-waste-api.hf.space
HF_SPACE_TIMEOUT_MS=55000
```

如果 Space 设置成公开访问，Vercel 不需要 `HF_SPACE_TOKEN`。如果后续改成受保护 Space，则需要：

```text
HF_SPACE_TOKEN=你的 Hugging Face runtime token
```

## OpenAI API 说明

项目仍保留 OpenAI Responses API 作为可选后端。后端会把上传图片或摄像头抽帧图片作为 `input_image` 发给模型，并要求模型直接给出泰国垃圾分类最终答案。

切换到 OpenAI：

```text
WASTE_CLASSIFIER_PROVIDER=openai
```

自动优先 Hugging Face，失败后回退 OpenAI：

```text
WASTE_CLASSIFIER_PROVIDER=auto
```

后端只做三件事：

1. 校验图片格式和大小。
2. 调用 OpenAI API。
3. 把 OpenAI 返回的结果整理成前端需要的 JSON。

后端不会再用本地关键词、像素颜色、评分机制或硬规则改写 OpenAI 的判断。

为了提高准确率，默认配置为：

```text
OPENAI_WASTE_MODEL=gpt-5.5
OPENAI_IMAGE_DETAIL=original
OPENAI_TIMEOUT_MS=45000
```

摄像头模式会发送较高分辨率的 JPEG 帧，减少透明塑料瓶、瓶盖、标签等细节丢失。

需要在 Vercel 环境变量中配置：

```text
OPENAI_API_KEY=你的 OpenAI API Key
OPENAI_BASE_URL=https://wokeme.dpdns.org/v1
```

可选环境变量：

```text
OPENAI_WASTE_MODEL=gpt-5.5
OPENAI_IMAGE_DETAIL=original
OPENAI_TIMEOUT_MS=45000
OPENAI_REASONING_EFFORT=
OPENAI_TEXT_VERBOSITY=
```

默认模型是：

```text
gpt-5.5
```

如果你想换成其他支持图片输入的 OpenAI 模型，只需要改 `OPENAI_WASTE_MODEL`。

官方文档参考：

```text
https://platform.openai.com/docs/guides/images-vision
https://platform.openai.com/docs/guides/structured-outputs
```

## 识别逻辑说明

当前默认版本不是 Vercel 本地模型推理，而是 Hugging Face Space 远程模型推理：

- Vercel 校验图片并转发到 Hugging Face Space。
- Space 加载完整不量化 ViT 模型。
- 模型输出 `green/yellow/blue/red` 和置信度。
- Vercel 映射为泰国/曼谷四桶分类建议。
- 前端展示分类、理由、置信度和投放建议。

如果 `WASTE_CLASSIFIER_PROVIDER=openai`，则会切换回 OpenAI 视觉问答。

## Hugging Face 模型仓库

项目里也已经训练好一个不量化的本地四桶分类模型，用于后续部署到 Hugging Face Model Repo 或独立推理服务：

```text
models/four-bin-waste-vit-v2-target-adapted/
```

模型类型：

```text
ViTForImageClassification
```

标签：

```text
green, yellow, blue, red
```

上传前先准备 Hugging Face Write token，然后运行：

```bash
export HF_TOKEN="你的 Hugging Face Write token"
.venv-ml/bin/python scripts/upload_hf_model_repo.py --dry-run
.venv-ml/bin/python scripts/upload_hf_model_repo.py
```

也可以先登录本地 Hugging Face CLI，再运行上传脚本：

```bash
.venv-ml/bin/hf auth login
.venv-ml/bin/hf auth whoami
.venv-ml/bin/python scripts/upload_hf_model_repo.py --dry-run
.venv-ml/bin/python scripts/upload_hf_model_repo.py
```

默认会上传到当前 Hugging Face 用户名下的 private model repo：

```text
bingo-thai-four-bin-waste-vit
```

如果要指定仓库名：

```bash
.venv-ml/bin/python scripts/upload_hf_model_repo.py \
  --repo-id "你的用户名/bingo-thai-four-bin-waste-vit"
```

如果要公开模型仓库：

```bash
.venv-ml/bin/python scripts/upload_hf_model_repo.py --public
```

上传脚本会检查本地模型文件，并在上传后验证远端仓库是否包含 `README.md`、`config.json`、`model.safetensors`、`preprocessor_config.json`、`thai_waste_labels.json` 等关键文件。

## 本地运行

如果只看静态页面，可以运行：

```bash
python3 -m http.server 4173
```

然后打开：

```text
http://localhost:4173/
```

如果要同时测试后端 `/api/classify`，建议使用 Vercel 本地开发服务：

```bash
vercel dev --listen 4174
```

然后打开：

```text
http://localhost:4174/
```

本地测试 OpenAI API 时，可以先设置环境变量：

```bash
export OPENAI_API_KEY="你的 OpenAI API Key"
export OPENAI_BASE_URL="https://wokeme.dpdns.org/v1"
export OPENAI_WASTE_MODEL="gpt-5.5"
export OPENAI_IMAGE_DETAIL="original"
vercel dev --listen 4174
```

## 摄像头说明

摄像头功能需要安全上下文：

- 线上 Vercel 地址是 HTTPS，可以正常请求摄像头权限。
- 本地 `localhost` 也可以使用摄像头。
- 普通 HTTP 域名可能无法打开摄像头。

手机上会优先请求后置摄像头：

```js
navigator.mediaDevices.getUserMedia({
  video: { facingMode: "environment" },
  audio: false
});
```

## 部署

当前生产地址：

```text
https://bingo-sorter.vercel.app/
```

部署命令：

```bash
vercel --prod
```

部署前先给 Vercel 设置环境变量：

```bash
vercel env add OPENAI_API_KEY production
vercel env add OPENAI_BASE_URL production
vercel env add OPENAI_WASTE_MODEL production
vercel env add OPENAI_IMAGE_DETAIL production
vercel env add OPENAI_TIMEOUT_MS production
```

`OPENAI_WASTE_MODEL` 可以填：

```text
gpt-5.5
```

`OPENAI_BASE_URL` 按 Codex 配置填：

```text
https://wokeme.dpdns.org/v1
```

如果需要重新绑定别名：

```bash
vercel alias set <deployment-url> bingo-sorter.vercel.app
```

## 文件说明

- `index.html`：页面结构，包括上传模式、摄像头模式、结果区域。
- `styles.css`：页面样式和响应式布局。
- `app.js`：前端交互、图片上传、摄像头控制、抽帧、调用后端、渲染结果。
- `api/classify.js`：Vercel Serverless API 入口。
- `api/lib/imageInput.js`：图片 base64、格式、大小和来源字段校验。
- `api/lib/openaiWasteClassifier.js`：OpenAI Responses API 图片问答和结构化结果整理。
- `api/lib/thaiWasteRules.js`：泰国垃圾桶展示文案、颜色、示例和投放建议。
- `assets/`：垃圾桶图片和页面标志图。

## 测试建议

上线后建议用手机打开：

```text
https://bingo-sorter.vercel.app/
```

测试流程：

1. 上传一张图片，确认可以返回分类结果。
2. 点击 Live camera。
3. 点击 Start camera 并允许摄像头权限。
4. 把物品放到扫描框中。
5. 等待自动分类，或点击 Classify now。
6. 点击 Stop camera，确认摄像头停止。
