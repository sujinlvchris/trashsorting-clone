# Thailand Waste Sorter Clone

Static single-page Thailand waste sorter with a Vercel backend classifier.

Public URL:

```text
https://bingo-sorter.vercel.app/
```

Run it locally:

```bash
python3 -m http.server 4173
```

Then open `http://localhost:4173/`.

The page implements image upload, drag-and-drop, preview, loading state, and a four-bin Thailand waste sorting result. Classification is handled by `POST /api/classify`, which uses a backend Thailand waste rules library and a lightweight image-feature scoring algorithm.

The current classifier follows the Greener Bangkok community waste sorting frame:

- Food / Organic waste
- Recyclable waste
- General waste
- Hazardous waste
