---
title: BinGo Thai Waste API
sdk: docker
app_port: 7860
pinned: false
---

# BinGo Thai Waste API

This Hugging Face Space serves the full-precision BinGo Thai four-bin waste classifier.

Endpoints:

- `GET /health`
- `POST /classify`

The Space loads:

```text
ChrisSujinlv/bingo-thai-four-bin-waste-vit
```

Required Space secret:

```text
HF_TOKEN
```

Optional Space variable:

```text
MODEL_REPO_ID=ChrisSujinlv/bingo-thai-four-bin-waste-vit
```
