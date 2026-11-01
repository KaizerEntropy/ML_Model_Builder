# CNN Forge

CNN Forge is a Vercel-ready React app for visually composing CNN architectures and classical machine-learning pipelines. It keeps the black-red research-console design, uses touch-friendly controls, and generates PyTorch or scikit-learn code from the current diagram.

## Run Locally

Install dependencies and start Vite:

```bash
npm install
npm run dev
```

Then open `http://localhost:3000`.

## Deploy To Vercel

```bash
vercel
```

Vercel will detect Vite, install dependencies, run `npm run build`, and serve the generated `dist` output. No Streamlit server is needed.

## Current Features

- Drag or tap blocks into the model diagram.
- Touch-friendly add, duplicate, reorder, delete, clear, and undo controls.
- CNN blocks: standard conv, 1x1 conv, 5x5 conv, dilated conv, depthwise separable conv, grouped conv, bottleneck, Inception, ASPP, ConvNeXt, ResNet, ResNet bottleneck, ResNeXt, MobileNet inverted residual, MBConv, Fused MBConv, ShuffleNet, Dense block, attention, normalization, pooling, dropout, flatten, GAP, and classifier head.
- Attention modules: SE, CBAM, ECA, coordinate attention, and self-attention.
- ResNet/ResNeXt blocks can draw skip connections to later blocks.
- Pooling options: max, average, adaptive max/average, GeM, and spatial pyramid pooling.
- Separate classical ML mode with random forest, decision tree, XGBoost, AdaBoost, logistic regression, linear regression, random forest regression, scalers, PCA, and image feature extraction placeholders.
- Per-block feature-map estimates for CNN diagrams.
- Dataset and augmentation panel.
- Download generated Python code, SVG diagram, or graph JSON.
- Run in Kaggle opens the complete generated code as a preloaded Python notebook; manual Python downloads remain available.

## Run in Kaggle

Deploy the app to Vercel, then click **Run in Kaggle**. The button snapshots the currently generated code, compresses it into a notebook source URL, verifies that the notebook endpoint responds, and opens Kaggle's [Load from URL flow](https://www.kaggle.com/discussions/product-feedback/152480). Sign into Kaggle if requested, then use **Run All** in the editor. This imports the code; it does not start training automatically or attach dataset files. Existing dataset loaders and advanced block placeholders still need implementation where marked in the generated code.

The `/api/notebook.ipynb` endpoint returns a standard Jupyter notebook with the exact code shown in the app, including your model, dataset references, augmentations, loss, and training settings. The generated code is encoded in the URL, which anyone holding the link can read. There is no notebook database or Kaggle API key to configure. Large notebooks that exceed the link limit show an error and can still be downloaded manually.

Kaggle must be able to fetch that endpoint over public HTTPS without deployment authentication. Localhost cannot be imported by Kaggle. For local development against an existing public deployment, optionally set `VITE_NOTEBOOK_BASE_URL=https://your-app.vercel.app` in `.env.local` and restart Vite. Otherwise use the deployed app for the Kaggle handoff. Vite dev and preview include the notebook endpoint for local API checks.

Run `npm test` to check notebook generation and import-link round trips across all five model modes.

## API Recommendation

For AI-assisted generation of new custom block code, use the OpenAI Responses API with Structured Outputs. Keep the UI graph as structured JSON, send the selected block schema plus constraints, and ask the model to return typed code fragments, validation notes, and shape metadata. This is better than free-form text because generated block code can be parsed, tested, and safely inserted into the PyTorch template.

## Dataset Link Notes

The app includes commonly used links for SIPaKMeD, Mendeley LBC Cervical Cancer, APCData, BACH, BreakHis, CIFAR-10, CIFAR-100, Caltech-101, and Food-101. Medical datasets can carry license, citation, and access requirements; verify those before training or publishing results.
