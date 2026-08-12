import { activations } from "../data/catalogs.js";

const Q = v => typeof v === "string" ? `"${v}"` : String(v);

function attExpr(att, ch) {
  if (att.kind === "se") return `SEBlock(${ch}, ratio=${att.params.ratio})`;
  if (att.kind === "cbam") return `CBAM(${ch}, ratio=${att.params.ratio})`;
  if (att.kind === "eca") return `ECABlock(${ch}, kernel_size=${att.params.kernel})`;
  if (att.kind === "coordattn") return `CoordinateAttention(${ch}, ratio=${att.params.ratio})`;
  if (att.kind === "selfattn") return `SelfAttention2d(${ch}, heads=${att.params.heads})`;
  if (att.kind === "gca") return `GlobalContextBlock(${ch}, ratio=${att.params.ratio})`;
  return "";
}

function pyDict(e) { return `{${Object.entries(e).map(([t, s]) => `${t}: [${s.join(", ")}]`).join(", ")}}`; }

// =======================
// TRAIN LOOP GENERATION
// =======================

function generatePyTorchTrainingLoop(mode, trainSettings, dataset) {
  const ts = trainSettings || {};
  const epochs = ts.epochs || 10;
  const bs = ts.batch_size || 32;
  const lr = ts.lr || 0.001;
  const split = ts.split || 0.2;
  const kfold = ts.kfold || 1;
  const opt = ts.optimizer || 'Adam';
  const vis = ts.visualizations || [];
  
  const isSeg = mode === "unet";
  const numCls = dataset.classes || 2;
  
  let code = `
# ==========================================
# Training & Evaluation Configuration
# ==========================================
import os
import copy
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

EPOCHS = ${epochs}
BATCH_SIZE = ${bs}
LEARNING_RATE = ${lr}
VAL_SPLIT = ${split}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example Dataset Placeholder (Replace with your actual data loader logic)
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.data = torch.randn(100, *MODEL_INPUT) # Dummy data
        ${isSeg ? `self.labels = torch.randint(0, 2, (100, 1, MODEL_INPUT[1], MODEL_INPUT[2])).float()` : `self.labels = torch.randint(0, ${numCls}, (100,))`}
    def __len__(self):
        return 100
    def __getitem__(self, idx):
        img = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

full_dataset = CustomDataset(transform=train_transforms if "train_transforms" in locals() else None)
val_size = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

model = model.to(DEVICE)
optimizer = optim.${opt}(model.parameters(), lr=LEARNING_RATE)

best_loss = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())
train_losses, val_losses = [], []

print(f"Starting Training on {DEVICE} for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    model.eval()
    val_loss = 0.0
    ${vis.includes("confusion_matrix") && !isSeg ? `all_preds = []\n    all_targets = []` : ""}
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
            ${vis.includes("confusion_matrix") && !isSeg ? `all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())\n            all_targets.extend(targets.cpu().numpy())` : ""}
    epoch_val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_train_loss:.4f} - Val Loss: {epoch_val_loss:.4f}")
    if epoch_val_loss < best_loss:
        best_loss = epoch_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())

model.load_state_dict(best_model_wts)
print("Training Complete. Best Val Loss: {:.4f}".format(best_loss))
`;

  if (vis.includes("plot_loss")) {
    code += `
# Plot Loss Curve
plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
`;
  }

  if (vis.includes("confusion_matrix") && !isSeg) {
    code += `
# Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(all_targets, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Validation)")
plt.show()
`;
  }

  if (vis.includes("tsne") && !isSeg) {
    code += `
# t-SNE Visualization (requires modifying model to output features)
# NOTE: Feature extraction depends on model architecture.
from sklearn.manifold import TSNE
print("t-SNE requested. Please ensure you extract the feature embeddings before the classification head.")
`;
  }
  return code;
}

// =======================
// ARCHITECTURE GEN
// =======================

export function generateTorchCode(graph, dataset, augs, lossConfig, trainSettings) {
  let c = dataset.shape[2], li = 0; const lm = new Map(), lines = [];
  graph.forEach(b => {
    const p = b.params; if (b.kind === "input") { c = Number(p.c || c); return; }
    let expr = "nn.Identity()", out = c;
    if (b.kind === "conv") { out = p.out; expr = `ConvBNAct(${c}, ${out}, ${p.kernel}, ${p.stride}, ${p.padding ?? 1}, ${p.dilation ?? 1}, ${Q(p.activation)})`; }
    else if (b.kind === "conv1x1") { out = p.out; expr = `ConvBNAct(${c}, ${out}, 1, 1, 0, 1, ${Q(p.activation)})`; }
    else if (b.kind === "conv5x5") { out = p.out; expr = `ConvBNAct(${c}, ${out}, 5, ${p.stride}, 2, 1, ${Q(p.activation)})`; }
    else if (b.kind === "dilated") { out = p.out; expr = `ConvBNAct(${c}, ${out}, ${p.kernel}, 1, ${p.dilation}, ${p.dilation}, ${Q(p.activation)})`; }
    else if (b.kind === "depthwise") { out = p.out; expr = `DepthwiseSeparableConv(${c}, ${out}, ${p.kernel}, ${p.stride}, ${Q(p.activation)})`; }
    else if (b.kind === "grouped") { out = p.out; expr = `GroupedConv(${c}, ${out}, groups=${p.groups}, kernel_size=${p.kernel}, stride=${p.stride}, activation=${Q(p.activation)})`; }
    else if (["bottleneck", "resnet_bottleneck"].includes(b.kind)) { out = p.out; expr = `BottleneckBlock(${c}, ${out}, bottleneck=${p.bottleneck}, stride=${p.stride}, activation=${Q(p.activation)})`; }
    else if (b.kind === "resnet") { out = p.out; expr = `ResidualBlock(${c}, ${out}, stride=${p.stride}, activation=${Q(p.activation)})`; }
    else if (b.kind === "resnext") { out = p.out; expr = `ResNeXtBlock(${c}, ${out}, cardinality=${p.cardinality}, stride=${p.stride}, activation=${Q(p.activation)})`; }
    else if (b.kind === "mobilenet") { out = p.out; expr = `InvertedResidual(${c}, ${out}, stride=${p.stride}, expansion=${p.expansion}, activation=${Q(p.activation)})`; }
    else if (b.kind === "efficientnet") { out = p.out; expr = `MBConv(${c}, ${out}, stride=${p.stride}, expansion=${p.expansion}, se_ratio=${p.se_ratio}, activation=${Q(p.activation)})`; }
    else if (b.kind === "fusedmb") { out = p.out; expr = `FusedMBConv(${c}, ${out}, stride=${p.stride}, expansion=${p.expansion}, activation=${Q(p.activation)})`; }
    else if (b.kind === "shuffle") { out = p.out; expr = `ShuffleUnit(${c}, ${out}, groups=${p.groups}, stride=${p.stride}, activation=${Q(p.activation)})`; }
    else if (b.kind === "inception") { out = p.out; expr = `InceptionBlock(${c}, ${out}, activation=${Q(p.activation)})`; }
    else if (b.kind === "aspp") { out = p.out; expr = `ASPPBlock(${c}, ${out}, activation=${Q(p.activation)})`; }
    else if (b.kind === "convnext") { out = p.out; expr = `ConvNeXtBlock(${c}, ${out}, kernel_size=${p.kernel}, activation=${Q(p.activation)})`; }
    else if (b.kind === "dense") { out = c + Number(p.growth) * Number(p.layers); expr = `DenseBlock(${c}, growth_rate=${p.growth}, num_layers=${p.layers}, activation=${Q(p.activation)})`; }
    else if (b.kind === "se") expr = `SEBlock(${c}, ratio=${p.ratio})`;
    else if (b.kind === "cbam") expr = `CBAM(${c}, ratio=${p.ratio})`;
    else if (b.kind === "eca") expr = `ECABlock(${c}, kernel_size=${p.kernel})`;
    else if (b.kind === "coordattn") expr = `CoordinateAttention(${c}, ratio=${p.ratio})`;
    else if (b.kind === "selfattn") expr = `SelfAttention2d(${c}, heads=${p.heads})`;
    else if (b.kind === "gca") expr = `GlobalContextBlock(${c}, ratio=${p.ratio})`;
    else if (b.kind === "activation") expr = `activation_layer(${Q(p.activation)})`;
    else if (b.kind === "batchnorm") expr = `nn.BatchNorm2d(${c})`;
    else if (b.kind === "groupnorm") expr = `nn.GroupNorm(${p.groups}, ${c})`;
    else if (b.kind === "instancenorm") expr = `nn.InstanceNorm2d(${c})`;
    else if (b.kind === "layernorm") expr = `nn.LayerNorm([${c}, H, W]) # Replace H,W with actual size`;
    else if (b.kind === "pool") expr = `nn.${p.mode}(kernel_size=${p.kernel}, stride=${p.stride})`;
    else if (b.kind === "avgpool") expr = `nn.AvgPool2d(kernel_size=${p.kernel}, stride=${p.stride})`;
    else if (b.kind === "adapool") expr = `nn.${p.mode}((${p.h}, ${p.w}))`;
    else if (b.kind === "gem") expr = `GeMPool(p=${p.p})`;
    else if (b.kind === "spp") expr = `SpatialPyramidPool([${p.bins}])`;
    else if (b.kind === "upsample") expr = `nn.Upsample(scale_factor=${p.scale}, mode=${Q(p.mode)})`;
    else if (b.kind === "transpose_conv") { out = p.out; expr = `nn.Sequential(nn.ConvTranspose2d(${c}, ${out}, ${p.kernel}, ${p.stride}, ${p.padding}), nn.BatchNorm2d(${out}), activation_layer(${Q(p.activation)}))`; }
    else if (b.kind === "pixelshuffle") { out = Math.max(1, Math.floor(c / (Number(p.scale) * Number(p.scale)))); expr = `nn.PixelShuffle(${p.scale})`; }
    else if (b.kind === "dropout") expr = `nn.Dropout(p=${p.p})`;
    else if (b.kind === "dropout2d") expr = `nn.Dropout2d(p=${p.p})`;
    else if (b.kind === "flatten") expr = "nn.Flatten()";
    else if (b.kind === "gap") expr = "nn.AdaptiveAvgPool2d((1, 1))";
    else if (b.kind === "head") { out = p.classes; expr = `ClassifierHead(${c}, num_classes=${p.classes}, hidden=${p.hidden}, dropout=${p.dropout})`; }
    
    const stack = [expr, ...(b.attention || []).map(a => attExpr(a, out))].filter(Boolean);
    lines.push(`            ("block_${li}_${b.kind}", ${stack.length > 1 ? `nn.Sequential(${stack.join(", ")})` : stack[0]}),`);
    lm.set(b.id, li); li++; c = out;
  });
  
  const se = {}; graph.forEach(b => {
    if (!b.params.skipTo) return;
    const s = lm.get(b.id), t = lm.get(b.params.skipTo);
    if (s === undefined || t === undefined || t <= s) return;
    se[t] ||= []; se[t].push(s);
  });
  const ss = [...new Set(Object.values(se).flat())].sort((a, b) => a - b);
  const lossLine = lossConfig ? `\ncriterion = ${generateLossExpr(lossConfig)}` : "";
  const trainLoopCode = generatePyTorchTrainingLoop("cnn", trainSettings, dataset);

  return `import torch, torch.nn as nn
from collections import OrderedDict
from torchvision import transforms

def activation_layer(name):
    return {"ReLU":nn.ReLU(True),"GELU":nn.GELU(),"SiLU":nn.SiLU(True),"Mish":nn.Mish(),"LeakyReLU":nn.LeakyReLU(0.1,True),"ReLU6":nn.ReLU6(True),"ELU":nn.ELU(True),"Hardswish":nn.Hardswish(),"PReLU":nn.PReLU(),"Tanh":nn.Tanh(),"Sigmoid":nn.Sigmoid()}.get(name, nn.ReLU(True))

# --- Building block classes (Placeholder for actual implementations) ---
# Ensure you have implementations for ConvBNAct, ResidualBlock, SEBlock, etc.
# in your local environment if utilizing advanced blocks.

class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(OrderedDict([
${lines.join("\n") || '            ("identity", nn.Identity()),'}
        ]))
        self.skip_edges = ${pyDict(se)}
        self.skip_sources = set(${JSON.stringify(ss)})

    def forward(self, x):
        saved = {}
        for i, block in enumerate(self.blocks):
            for si in self.skip_edges.get(i, []):
                x = x + self._match(saved[si], x)
            x = block(x)
            if i in self.skip_sources:
                saved[i] = x
        return x

    def _match(self, skip, x):
        if skip.shape[-2:] != x.shape[-2:]:
            skip = torch.nn.functional.interpolate(skip, size=x.shape[-2:], mode="nearest")
        if skip.shape[1] < x.shape[1]:
            skip = torch.cat([skip, skip.new_zeros(skip.shape[0], x.shape[1]-skip.shape[1], *skip.shape[2:])], 1)
        elif skip.shape[1] > x.shape[1]:
            skip = skip[:, :x.shape[1]]
        return skip

${genTransforms(dataset, augs)}

DATASET_NAME = "${dataset.name}"
MODEL_INPUT = (${dataset.shape[2]}, ${dataset.shape[0]}, ${dataset.shape[1]})
${lossLine}
model = CustomCNN()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
${trainLoopCode}
`;
}

function genTransforms(ds, augs) {
  const [h, w] = ds.shape, l = [];
  if (augs.includes("Resize")) l.push(`    transforms.Resize((${h}, ${w})),`);
  if (augs.includes("RandomResizedCrop")) l.push(`    transforms.RandomResizedCrop(${Math.max(h, w)}),`);
  if (augs.includes("HorizontalFlip")) l.push("    transforms.RandomHorizontalFlip(p=0.5),");
  if (augs.includes("VerticalFlip")) l.push("    transforms.RandomVerticalFlip(p=0.2),");
  if (augs.includes("Rotation")) l.push("    transforms.RandomRotation(20),");
  if (augs.includes("ColorJitter")) l.push("    transforms.ColorJitter(0.2, 0.2, 0.15, 0.03),");
  if (augs.includes("GaussianBlur")) l.push("    transforms.GaussianBlur(3),");
  if (augs.includes("AutoAugment")) l.push("    transforms.AutoAugment(),");
  if (augs.includes("RandAugment")) l.push("    transforms.RandAugment(),");
  l.push("    transforms.ToTensor(),");
  if (augs.includes("Normalize")) l.push("    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),");
  if (augs.includes("RandomErasing")) l.push("    transforms.RandomErasing(p=0.25),");
  return `train_transforms = transforms.Compose([\n${l.join("\n")}\n])\nUSE_MIXUP = ${augs.includes("MixUp")}\nUSE_CUTMIX = ${augs.includes("CutMix")}`;
}

function generateLossExpr(cfg) {
  const p = cfg.params || {};
  const m = { cross_entropy: `nn.CrossEntropyLoss(label_smoothing=${p.label_smoothing || 0.0})`, bce: "nn.BCEWithLogitsLoss()", focal: `FocalLoss(alpha=${p.alpha || 0.25}, gamma=${p.gamma || 2.0})`, mse: "nn.MSELoss()", l1: "nn.L1Loss()", smooth_l1: `nn.SmoothL1Loss(beta=${p.beta || 1.0})`, kl_div: "nn.KLDivLoss(reduction='batchmean')", triplet: `nn.TripletMarginLoss(margin=${p.margin || 1.0})`, contrastive: `ContrastiveLoss(margin=${p.margin || 1.0})`, cosine_embed: `nn.CosineEmbeddingLoss(margin=${p.margin || 0.0})`, dice: `DiceLoss(smooth=${p.smooth || 1.0})`, bce_dice: `BCEDiceLoss(bce_weight=${p.bce_weight || 0.5})`, tversky: `TverskyLoss(alpha=${p.alpha || 0.3}, beta=${p.beta || 0.7})`, lovasz: "LovaszSoftmaxLoss()", boundary: "BoundaryLoss()", hausdorff: "HausdorffLoss()", ctc: `nn.CTCLoss(blank=${p.blank || 0})` };
  return m[cfg.kind] || "nn.CrossEntropyLoss()";
}

export function generateTransformerCode(graph, dataset, augs, lossConfig, trainSettings) {
  const lines = []; let dim = 768, li = 0;
  graph.forEach(b => {
    const p = b.params; if (b.kind === "t_input") return;
    let expr = "nn.Identity()";
    if (b.kind === "patch_embed") expr = `PatchEmbedding(img_size=${p.img_size}, patch_size=${p.patch_size}, in_channels=${p.in_channels}, embed_dim=${p.embed_dim})`;
    else if (b.kind === "pos_embed") expr = `PositionalEncoding(max_len=${p.max_len}, embed_dim=${dim}, mode=${Q(p.mode)})`;
    else if (b.kind === "cls_token") expr = `CLSToken(embed_dim=${dim})`;
    else if (["mha", "cross_attn", "masked_mha"].includes(b.kind)) { dim = Number(p.embed_dim || dim); expr = `nn.MultiheadAttention(${dim}, ${p.heads}, dropout=${p.dropout}, batch_first=True)`; }
    else if (b.kind === "ffn") { dim = Number(p.embed_dim || dim); expr = `FeedForward(${dim}, ${p.hidden}, activation=${Q(p.activation)}, dropout=${p.dropout})`; }
    else if (["transformer_block", "vit_block", "deit_block"].includes(b.kind)) { dim = Number(p.embed_dim || dim); expr = `TransformerBlock(${dim}, heads=${p.heads}, ffn_dim=${p.hidden || dim * 4}, dropout=${p.dropout || 0.0})`; }
    else if (b.kind === "swin_block") { dim = Number(p.embed_dim || dim); expr = `SwinTransformerBlock(${dim}, heads=${p.heads}, window_size=${p.window_size})`; }
    else if (b.kind === "decoder_block") { dim = Number(p.embed_dim || dim); expr = `TransformerDecoderBlock(${dim}, heads=${p.heads}, ffn_dim=${p.hidden || dim * 4}, dropout=${p.dropout || 0.0})`; }
    else if (b.kind === "t_layernorm") expr = `nn.LayerNorm(${dim})`;
    else if (b.kind === "t_dropout") expr = `nn.Dropout(p=${p.p})`;
    else if (b.kind === "t_cls_pool") expr = "CLSPooling()";
    else if (b.kind === "t_mean_pool") expr = "MeanPooling()";
    else if (b.kind === "t_head") { expr = `ClassifierHead(${dim}, num_classes=${p.classes}, hidden=${p.hidden}, dropout=${p.dropout})`; dim = Number(p.classes); }
    lines.push(`            ("block_${li}_${b.kind}", ${expr}),`); li++;
  });
  
  const lossLine = lossConfig ? `\ncriterion = ${generateLossExpr(lossConfig)}` : "";
  const trainLoopCode = generatePyTorchTrainingLoop("transformer", trainSettings, dataset);

  return `import torch, torch.nn as nn
from collections import OrderedDict
from torchvision import transforms

class CustomTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(OrderedDict([
${lines.join("\n") || '            ("identity", nn.Identity()),'}
        ]))
    def forward(self, x):
        return self.blocks(x)

${genTransforms(dataset, augs)}
DATASET_NAME = "${dataset.name}"
MODEL_INPUT = (${dataset.shape[2]}, ${dataset.shape[0]}, ${dataset.shape[1]})
${lossLine}
model = CustomTransformer()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
${trainLoopCode}
`;
}

export function generateUnetCode(graph, dataset, augs, lossConfig, trainSettings) {
  const enc = [], dec = [], bot = [], head = []; let c = dataset.shape[2];
  graph.forEach(b => {
    const p = b.params;
    if (b.kind === "u_input") return;
    if (b.kind.startsWith("u_enc")) { enc.push(`        self.enc${enc.length} = EncoderBlock(${c}, ${p.out}, activation=${Q(p.activation || "ReLU")})`); c = Number(p.out); }
    else if (["u_bottleneck", "u_aspp_bridge"].includes(b.kind)) { bot.push(`        self.bottleneck = BottleneckBridge(${c}, ${p.out}, activation=${Q(p.activation || "ReLU")})`); c = Number(p.out); }
    else if (b.kind.startsWith("u_dec")) { dec.push(`        self.dec${dec.length} = DecoderBlock(${c}, ${p.out}, activation=${Q(p.activation || "ReLU")})`); c = Number(p.out); }
    else if (b.kind === "u_seg_head") { head.push(`        self.seg_head = nn.Conv2d(${c}, ${p.classes}, 1)`); }
    else if (b.kind === "u_attn_gate") { head.push(`        self.attn_gate = AttentionGate(${p.channels || c})`); }
  });
  const lossLine = lossConfig ? `\ncriterion = ${generateLossExpr(lossConfig)}` : "";
  const trainLoopCode = generatePyTorchTrainingLoop("unet", trainSettings, dataset);

  return `import torch, torch.nn as nn
from torchvision import transforms

class CustomUNet(nn.Module):
    def __init__(self):
        super().__init__()
${enc.join("\n")}
${bot.join("\n")}
${dec.join("\n")}
${head.join("\n")}

    def forward(self, x):
        skips = []
${enc.map((_, i) => `        x = self.enc${i}(x); skips.append(x)`).join("\n")}
${bot.length ? `        x = self.bottleneck(x)` : "        pass"}
${dec.map((_, i) => `        x = self.dec${i}(x, skips[${enc.length - 1 - i}] if ${enc.length - 1 - i} < len(skips) else None)`).join("\n")}
${head.length ? `        x = self.seg_head(x)` : "        pass"}
        return x

${genTransforms(dataset, augs)}
DATASET_NAME = "${dataset.name}"
MODEL_INPUT = (${dataset.shape[2]}, ${dataset.shape[0]}, ${dataset.shape[1]})
${lossLine}
model = CustomUNet()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
${trainLoopCode}
`;
}

export function generateSklearnCode(graph, lossConfig, trainSettings) {
  const imp = new Set(["from sklearn.pipeline import Pipeline", "from sklearn.model_selection import train_test_split", "from sklearn.metrics import classification_report", "import matplotlib.pyplot as plt"]);
  const steps = [];
  const ts = trainSettings || {};
  const vis = ts.visualizations || [];
  const testSplit = ts.split || 0.2;

  graph.forEach(b => {
    const p = b.params;
    if (b.kind === "standard_scaler") { imp.add("from sklearn.preprocessing import StandardScaler"); steps.push('    ("scaler", StandardScaler()),'); }
    else if (b.kind === "minmax_scaler") { imp.add("from sklearn.preprocessing import MinMaxScaler"); steps.push('    ("scaler", MinMaxScaler()),'); }
    else if (b.kind === "robust_scaler") { imp.add("from sklearn.preprocessing import RobustScaler"); steps.push('    ("scaler", RobustScaler()),'); }
    else if (b.kind === "pca") { imp.add("from sklearn.decomposition import PCA"); steps.push(`    ("pca", PCA(n_components=${p.components})),`); }
    else if (b.kind === "imputer") { imp.add("from sklearn.impute import SimpleImputer"); steps.push(`    ("imputer", SimpleImputer(strategy=${Q(p.strategy || "mean")})),`); }
    else if (b.kind === "random_forest") { imp.add("from sklearn.ensemble import RandomForestClassifier"); steps.push(`    ("model", RandomForestClassifier(n_estimators=${p.n_estimators}, max_depth=${p.max_depth}, random_state=42)),`); }
    else if (b.kind === "decision_tree") { imp.add("from sklearn.tree import DecisionTreeClassifier"); steps.push(`    ("model", DecisionTreeClassifier(max_depth=${p.max_depth}, random_state=42)),`); }
    else if (b.kind === "xgboost") { imp.add("from xgboost import XGBClassifier"); steps.push(`    ("model", XGBClassifier(n_estimators=${p.n_estimators}, learning_rate=${p.learning_rate}, max_depth=${p.max_depth})),`); }
    else if (b.kind === "adaboost") { imp.add("from sklearn.ensemble import AdaBoostClassifier"); steps.push(`    ("model", AdaBoostClassifier(n_estimators=${p.n_estimators}, learning_rate=${p.learning_rate})),`); }
    else if (b.kind === "logistic_regression") { imp.add("from sklearn.linear_model import LogisticRegression"); steps.push(`    ("model", LogisticRegression(C=${p.C}, max_iter=${p.max_iter})),`); }
    else if (b.kind === "svm") { imp.add("from sklearn.svm import SVC"); steps.push(`    ("model", SVC(C=${p.C}, kernel=${Q(p.kernel)})),`); }
    else if (b.kind === "knn") { imp.add("from sklearn.neighbors import KNeighborsClassifier"); steps.push(`    ("model", KNeighborsClassifier(n_neighbors=${p.n_neighbors})),`); }
    else if (b.kind === "naive_bayes") { imp.add("from sklearn.naive_bayes import GaussianNB"); steps.push('    ("model", GaussianNB()),'); }
    else if (b.kind === "gradient_boosting") { imp.add("from sklearn.ensemble import GradientBoostingClassifier"); steps.push(`    ("model", GradientBoostingClassifier(n_estimators=${p.n_estimators}, learning_rate=${p.learning_rate}, max_depth=${p.max_depth})),`); }
    else if (b.kind === "extra_trees") { imp.add("from sklearn.ensemble import ExtraTreesClassifier"); steps.push(`    ("model", ExtraTreesClassifier(n_estimators=${p.n_estimators}, max_depth=${p.max_depth})),`); }
  });

  let extras = "";
  if (vis.includes("confusion_matrix")) {
      imp.add("from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay");
      extras += `
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
`;
  }
  
  return `${[...imp].sort().join("\n")}
import pandas as pd

DATA_PATH = "data.csv"
TARGET = "label" # Replace with actual target column
try:
    df = pd.read_csv(DATA_PATH)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=${testSplit}, random_state=42)

    pipeline = Pipeline([
${steps.join("\n") || '    ("model", RandomForestClassifier(n_estimators=300, random_state=42)),'}
    ])

    print("Fitting model...")
    pipeline.fit(X_train, y_train)
    
    print("Evaluating model...")
    preds = pipeline.predict(X_test)
    print(classification_report(y_test, preds))
${extras}
except Exception as e:
    print("Dataset not found or target column missing. Please ensure data.csv exists and TARGET is set correctly.")
`;
}
