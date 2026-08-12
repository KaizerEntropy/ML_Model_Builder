import {activations} from "../data/catalogs.js";
const Q=v=>typeof v==="string"?`"${v}"`:String(v);
function attExpr(att,ch){
  if(att.kind==="se")return`SEBlock(${ch}, ratio=${att.params.ratio})`;
  if(att.kind==="cbam")return`CBAM(${ch}, ratio=${att.params.ratio})`;
  if(att.kind==="eca")return`ECABlock(${ch}, kernel_size=${att.params.kernel})`;
  if(att.kind==="coordattn")return`CoordinateAttention(${ch}, ratio=${att.params.ratio})`;
  if(att.kind==="selfattn")return`SelfAttention2d(${ch}, heads=${att.params.heads})`;
  if(att.kind==="gca")return`GlobalContextBlock(${ch}, ratio=${att.params.ratio})`;
  return"";
}
function pyDict(e){return`{${Object.entries(e).map(([t,s])=>`${t}: [${s.join(", ")}]`).join(", ")}}`;}

export function generateTorchCode(graph,dataset,augs,lossConfig){
  let c=dataset.shape[2],li=0;const lm=new Map(),lines=[];
  graph.forEach(b=>{
    const p=b.params;if(b.kind==="input"){c=Number(p.c||c);return;}
    let expr="nn.Identity()",out=c;
    if(b.kind==="conv"){out=p.out;expr=`ConvBNAct(${c}, ${out}, ${p.kernel}, ${p.stride}, ${p.padding??1}, ${p.dilation??1}, ${Q(p.activation)})`;}
    if(b.kind==="conv1x1"){out=p.out;expr=`ConvBNAct(${c}, ${out}, 1, 1, 0, 1, ${Q(p.activation)})`;}
    if(b.kind==="conv5x5"){out=p.out;expr=`ConvBNAct(${c}, ${out}, 5, ${p.stride}, 2, 1, ${Q(p.activation)})`;}
    if(b.kind==="dilated"){out=p.out;expr=`ConvBNAct(${c}, ${out}, ${p.kernel}, 1, ${p.dilation}, ${p.dilation}, ${Q(p.activation)})`;}
    if(b.kind==="depthwise"){out=p.out;expr=`DepthwiseSeparableConv(${c}, ${out}, ${p.kernel}, ${p.stride}, ${Q(p.activation)})`;}
    if(b.kind==="grouped"){out=p.out;expr=`GroupedConv(${c}, ${out}, groups=${p.groups}, kernel_size=${p.kernel}, stride=${p.stride}, activation=${Q(p.activation)})`;}
    if(["bottleneck","resnet_bottleneck"].includes(b.kind)){out=p.out;expr=`BottleneckBlock(${c}, ${out}, bottleneck=${p.bottleneck}, stride=${p.stride}, activation=${Q(p.activation)})`;}
    if(b.kind==="resnet"){out=p.out;expr=`ResidualBlock(${c}, ${out}, stride=${p.stride}, activation=${Q(p.activation)})`;}
    if(b.kind==="resnext"){out=p.out;expr=`ResNeXtBlock(${c}, ${out}, cardinality=${p.cardinality}, stride=${p.stride}, activation=${Q(p.activation)})`;}
    if(b.kind==="mobilenet"){out=p.out;expr=`InvertedResidual(${c}, ${out}, stride=${p.stride}, expansion=${p.expansion}, activation=${Q(p.activation)})`;}
    if(b.kind==="efficientnet"){out=p.out;expr=`MBConv(${c}, ${out}, stride=${p.stride}, expansion=${p.expansion}, se_ratio=${p.se_ratio}, activation=${Q(p.activation)})`;}
    if(b.kind==="fusedmb"){out=p.out;expr=`FusedMBConv(${c}, ${out}, stride=${p.stride}, expansion=${p.expansion}, activation=${Q(p.activation)})`;}
    if(b.kind==="shuffle"){out=p.out;expr=`ShuffleUnit(${c}, ${out}, groups=${p.groups}, stride=${p.stride}, activation=${Q(p.activation)})`;}
    if(b.kind==="inception"){out=p.out;expr=`InceptionBlock(${c}, ${out}, activation=${Q(p.activation)})`;}
    if(b.kind==="aspp"){out=p.out;expr=`ASPPBlock(${c}, ${out}, activation=${Q(p.activation)})`;}
    if(b.kind==="convnext"){out=p.out;expr=`ConvNeXtBlock(${c}, ${out}, kernel_size=${p.kernel}, activation=${Q(p.activation)})`;}
    if(b.kind==="dense"){out=c+Number(p.growth)*Number(p.layers);expr=`DenseBlock(${c}, growth_rate=${p.growth}, num_layers=${p.layers}, activation=${Q(p.activation)})`;}
    if(b.kind==="se")expr=`SEBlock(${c}, ratio=${p.ratio})`;
    if(b.kind==="cbam")expr=`CBAM(${c}, ratio=${p.ratio})`;
    if(b.kind==="eca")expr=`ECABlock(${c}, kernel_size=${p.kernel})`;
    if(b.kind==="coordattn")expr=`CoordinateAttention(${c}, ratio=${p.ratio})`;
    if(b.kind==="selfattn")expr=`SelfAttention2d(${c}, heads=${p.heads})`;
    if(b.kind==="gca")expr=`GlobalContextBlock(${c}, ratio=${p.ratio})`;
    if(b.kind==="activation")expr=`activation_layer(${Q(p.activation)})`;
    if(b.kind==="batchnorm")expr=`nn.BatchNorm2d(${c})`;
    if(b.kind==="groupnorm")expr=`nn.GroupNorm(${p.groups}, ${c})`;
    if(b.kind==="instancenorm")expr=`nn.InstanceNorm2d(${c})`;
    if(b.kind==="layernorm")expr=`nn.LayerNorm(${c})`;
    if(b.kind==="pool")expr=`nn.${p.mode}(kernel_size=${p.kernel}, stride=${p.stride})`;
    if(b.kind==="avgpool")expr=`nn.AvgPool2d(kernel_size=${p.kernel}, stride=${p.stride})`;
    if(b.kind==="adapool")expr=`nn.${p.mode}((${p.h}, ${p.w}))`;
    if(b.kind==="gem")expr=`GeMPool(p=${p.p})`;
    if(b.kind==="spp")expr=`SpatialPyramidPool([${p.bins}])`;
    if(b.kind==="upsample")expr=`nn.Upsample(scale_factor=${p.scale}, mode=${Q(p.mode)})`;
    if(b.kind==="transpose_conv"){out=p.out;expr=`nn.Sequential(nn.ConvTranspose2d(${c}, ${out}, ${p.kernel}, ${p.stride}, ${p.padding}), nn.BatchNorm2d(${out}), activation_layer(${Q(p.activation)}))`;}
    if(b.kind==="pixelshuffle"){out=Math.max(1,Math.floor(c/(Number(p.scale)*Number(p.scale))));expr=`nn.PixelShuffle(${p.scale})`;}
    if(b.kind==="dropout")expr=`nn.Dropout(p=${p.p})`;
    if(b.kind==="dropout2d")expr=`nn.Dropout2d(p=${p.p})`;
    if(b.kind==="flatten")expr="nn.Flatten()";
    if(b.kind==="gap")expr="nn.AdaptiveAvgPool2d((1, 1))";
    if(b.kind==="head"){out=p.classes;expr=`ClassifierHead(${c}, num_classes=${p.classes}, hidden=${p.hidden}, dropout=${p.dropout})`;}
    const stack=[expr,...(b.attention||[]).map(a=>attExpr(a,out))].filter(Boolean);
    lines.push(`            ("block_${li}_${b.kind}", ${stack.length>1?`nn.Sequential(${stack.join(", ")})`:stack[0]}),`);
    lm.set(b.id,li);li++;c=out;
  });
  const se={};graph.forEach(b=>{if(!b.params.skipTo)return;const s=lm.get(b.id),t=lm.get(b.params.skipTo);if(s===undefined||t===undefined||t<=s)return;se[t]||=[];se[t].push(s);});
  const ss=[...new Set(Object.values(se).flat())].sort((a,b)=>a-b);
  const lossLine=lossConfig?`\ncriterion = ${generateLossExpr(lossConfig)}`:"";
  return `import torch, torch.nn as nn
from collections import OrderedDict
from torchvision import transforms

def activation_layer(name):
    return {"ReLU":nn.ReLU(True),"GELU":nn.GELU(),"SiLU":nn.SiLU(True),"Mish":nn.Mish(),"LeakyReLU":nn.LeakyReLU(0.1,True),"ReLU6":nn.ReLU6(True),"ELU":nn.ELU(True),"Hardswish":nn.Hardswish(),"PReLU":nn.PReLU(),"Tanh":nn.Tanh(),"Sigmoid":nn.Sigmoid()}.get(name, nn.ReLU(True))

# --- Building block classes (see previous definitions) ---
# ConvBNAct, DepthwiseSeparableConv, GroupedConv, SEBlock, ECABlock, CBAM,
# CoordinateAttention, SelfAttention2d, GlobalContextBlock, ResidualBlock,
# BottleneckBlock, ResNeXtBlock, InvertedResidual, MBConv, FusedMBConv,
# ShuffleUnit, InceptionBlock, ASPPBlock, ConvNeXtBlock, DenseBlock,
# GeMPool, SpatialPyramidPool, ClassifierHead

class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(OrderedDict([
${lines.join("\n")||'            ("identity", nn.Identity()),'}
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

${genTransforms(dataset,augs)}

DATASET_NAME = "${dataset.name}"
DATASET_URL = "${dataset.url}"
MODEL_INPUT = (${dataset.shape[2]}, ${dataset.shape[0]}, ${dataset.shape[1]})
${lossLine}
model = CustomCNN()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
`;
}

function genTransforms(ds,augs){
  const[h,w]=ds.shape,l=[];
  if(augs.includes("Resize"))l.push(`    transforms.Resize((${h}, ${w})),`);
  if(augs.includes("RandomResizedCrop"))l.push(`    transforms.RandomResizedCrop(${Math.max(h,w)}),`);
  if(augs.includes("HorizontalFlip"))l.push("    transforms.RandomHorizontalFlip(p=0.5),");
  if(augs.includes("VerticalFlip"))l.push("    transforms.RandomVerticalFlip(p=0.2),");
  if(augs.includes("Rotation"))l.push("    transforms.RandomRotation(20),");
  if(augs.includes("ColorJitter"))l.push("    transforms.ColorJitter(0.2, 0.2, 0.15, 0.03),");
  if(augs.includes("GaussianBlur"))l.push("    transforms.GaussianBlur(3),");
  if(augs.includes("AutoAugment"))l.push("    transforms.AutoAugment(),");
  if(augs.includes("RandAugment"))l.push("    transforms.RandAugment(),");
  l.push("    transforms.ToTensor(),");
  if(augs.includes("Normalize"))l.push("    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),");
  if(augs.includes("RandomErasing"))l.push("    transforms.RandomErasing(p=0.25),");
  return`train_transforms = transforms.Compose([\n${l.join("\n")}\n])\nUSE_MIXUP = ${augs.includes("MixUp")}\nUSE_CUTMIX = ${augs.includes("CutMix")}`;
}

function generateLossExpr(cfg){
  const p=cfg.params||{};
  const m={cross_entropy:`nn.CrossEntropyLoss(label_smoothing=${p.label_smoothing||0.0})`,bce:"nn.BCEWithLogitsLoss()",focal:`FocalLoss(alpha=${p.alpha||0.25}, gamma=${p.gamma||2.0})`,mse:"nn.MSELoss()",l1:"nn.L1Loss()",smooth_l1:`nn.SmoothL1Loss(beta=${p.beta||1.0})`,kl_div:"nn.KLDivLoss(reduction='batchmean')",triplet:`nn.TripletMarginLoss(margin=${p.margin||1.0})`,contrastive:`ContrastiveLoss(margin=${p.margin||1.0})`,cosine_embed:`nn.CosineEmbeddingLoss(margin=${p.margin||0.0})`,dice:`DiceLoss(smooth=${p.smooth||1.0})`,bce_dice:`BCEDiceLoss(bce_weight=${p.bce_weight||0.5})`,tversky:`TverskyLoss(alpha=${p.alpha||0.3}, beta=${p.beta||0.7})`,lovasz:"LovaszSoftmaxLoss()",boundary:"BoundaryLoss()",hausdorff:"HausdorffLoss()",ctc:`nn.CTCLoss(blank=${p.blank||0})`};
  return m[cfg.kind]||"nn.CrossEntropyLoss()";
}

export function generateTransformerCode(graph,dataset,augs,lossConfig){
  const lines=[];let dim=768,li=0;
  graph.forEach(b=>{
    const p=b.params;if(["t_input"].includes(b.kind))return;
    let expr="nn.Identity()";
    if(b.kind==="patch_embed")expr=`PatchEmbedding(img_size=${p.img_size}, patch_size=${p.patch_size}, in_channels=${p.in_channels}, embed_dim=${p.embed_dim})`;
    if(b.kind==="pos_embed")expr=`PositionalEncoding(max_len=${p.max_len}, embed_dim=${dim}, mode=${Q(p.mode)})`;
    if(b.kind==="cls_token")expr=`CLSToken(embed_dim=${dim})`;
    if(["mha","cross_attn","masked_mha"].includes(b.kind)){dim=Number(p.embed_dim||dim);expr=`nn.MultiheadAttention(${dim}, ${p.heads}, dropout=${p.dropout}, batch_first=True)`;}
    if(b.kind==="ffn"){dim=Number(p.embed_dim||dim);expr=`FeedForward(${dim}, ${p.hidden}, activation=${Q(p.activation)}, dropout=${p.dropout})`;}
    if(["transformer_block","vit_block","deit_block"].includes(b.kind)){dim=Number(p.embed_dim||dim);expr=`TransformerBlock(${dim}, heads=${p.heads}, ffn_dim=${p.hidden||dim*4}, dropout=${p.dropout||0.0})`;}
    if(b.kind==="swin_block"){dim=Number(p.embed_dim||dim);expr=`SwinTransformerBlock(${dim}, heads=${p.heads}, window_size=${p.window_size})`;}
    if(b.kind==="decoder_block"){dim=Number(p.embed_dim||dim);expr=`TransformerDecoderBlock(${dim}, heads=${p.heads}, ffn_dim=${p.hidden||dim*4}, dropout=${p.dropout||0.0})`;}
    if(b.kind==="t_layernorm")expr=`nn.LayerNorm(${dim})`;
    if(b.kind==="t_dropout")expr=`nn.Dropout(p=${p.p})`;
    if(b.kind==="t_cls_pool")expr="CLSPooling()";
    if(b.kind==="t_mean_pool")expr="MeanPooling()";
    if(b.kind==="t_head"){expr=`ClassifierHead(${dim}, num_classes=${p.classes}, hidden=${p.hidden}, dropout=${p.dropout})`;dim=Number(p.classes);}
    lines.push(`            ("block_${li}_${b.kind}", ${expr}),`);li++;
  });
  const lossLine=lossConfig?`\ncriterion = ${generateLossExpr(lossConfig)}`:"";
  return `import torch, torch.nn as nn
from collections import OrderedDict

# Transformer building blocks: PatchEmbedding, PositionalEncoding, CLSToken,
# FeedForward, TransformerBlock, SwinTransformerBlock, TransformerDecoderBlock,
# CLSPooling, MeanPooling, ClassifierHead

class CustomTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(OrderedDict([
${lines.join("\n")||'            ("identity", nn.Identity()),'}
        ]))

    def forward(self, x):
        return self.blocks(x)

DATASET_NAME = "${dataset.name}"
MODEL_INPUT = (${dataset.shape[2]}, ${dataset.shape[0]}, ${dataset.shape[1]})
${lossLine}
model = CustomTransformer()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
`;
}

export function generateUnetCode(graph,dataset,augs,lossConfig){
  const enc=[],dec=[],bot=[],head=[];let c=dataset.shape[2];
  graph.forEach(b=>{
    const p=b.params;
    if(b.kind==="u_input")return;
    if(b.kind.startsWith("u_enc")){enc.push(`        self.enc${enc.length} = EncoderBlock(${c}, ${p.out}, activation=${Q(p.activation||"ReLU")})`);c=Number(p.out);}
    else if(["u_bottleneck","u_aspp_bridge"].includes(b.kind)){bot.push(`        self.bottleneck = BottleneckBridge(${c}, ${p.out}, activation=${Q(p.activation||"ReLU")})`);c=Number(p.out);}
    else if(b.kind.startsWith("u_dec")){dec.push(`        self.dec${dec.length} = DecoderBlock(${c}, ${p.out}, activation=${Q(p.activation||"ReLU")})`);c=Number(p.out);}
    else if(b.kind==="u_seg_head"){head.push(`        self.seg_head = nn.Conv2d(${c}, ${p.classes}, 1)`);}
    else if(b.kind==="u_attn_gate"){head.push(`        self.attn_gate = AttentionGate(${p.channels||c})`);}
  });
  const lossLine=lossConfig?`\ncriterion = ${generateLossExpr(lossConfig)}`:"";
  return `import torch, torch.nn as nn

# U-Net building blocks: EncoderBlock, DecoderBlock, BottleneckBridge,
# AttentionGate, SegmentationHead

class CustomUNet(nn.Module):
    def __init__(self):
        super().__init__()
${enc.join("\n")}
${bot.join("\n")}
${dec.join("\n")}
${head.join("\n")}

    def forward(self, x):
        # Encoder path with skip connections
        skips = []
${enc.map((_,i)=>`        x = self.enc${i}(x); skips.append(x)`).join("\n")}
        # Bottleneck
${bot.length?`        x = self.bottleneck(x)`:"        pass"}
        # Decoder path
${dec.map((_,i)=>`        x = self.dec${i}(x, skips[${enc.length-1-i}] if ${enc.length-1-i} < len(skips) else None)`).join("\n")}
        # Head
${head.length?`        x = self.seg_head(x)`:"        pass"}
        return x

DATASET_NAME = "${dataset.name}"
MODEL_INPUT = (${dataset.shape[2]}, ${dataset.shape[0]}, ${dataset.shape[1]})
${lossLine}
model = CustomUNet()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
`;
}

export function generateSklearnCode(graph,lossConfig){
  const imp=new Set(["from sklearn.pipeline import Pipeline","from sklearn.model_selection import train_test_split","from sklearn.metrics import classification_report"]);
  const steps=[];
  graph.forEach(b=>{const p=b.params;
    if(b.kind==="standard_scaler"){imp.add("from sklearn.preprocessing import StandardScaler");steps.push('    ("scaler", StandardScaler()),');}
    if(b.kind==="minmax_scaler"){imp.add("from sklearn.preprocessing import MinMaxScaler");steps.push('    ("scaler", MinMaxScaler()),');}
    if(b.kind==="robust_scaler"){imp.add("from sklearn.preprocessing import RobustScaler");steps.push('    ("scaler", RobustScaler()),');}
    if(b.kind==="pca"){imp.add("from sklearn.decomposition import PCA");steps.push(`    ("pca", PCA(n_components=${p.components})),`);}
    if(b.kind==="imputer"){imp.add("from sklearn.impute import SimpleImputer");steps.push(`    ("imputer", SimpleImputer(strategy=${Q(p.strategy||"mean")})),`);}
    if(b.kind==="random_forest"){imp.add("from sklearn.ensemble import RandomForestClassifier");steps.push(`    ("model", RandomForestClassifier(n_estimators=${p.n_estimators}, max_depth=${p.max_depth}, random_state=42)),`);}
    if(b.kind==="decision_tree"){imp.add("from sklearn.tree import DecisionTreeClassifier");steps.push(`    ("model", DecisionTreeClassifier(max_depth=${p.max_depth}, random_state=42)),`);}
    if(b.kind==="xgboost"){imp.add("from xgboost import XGBClassifier");steps.push(`    ("model", XGBClassifier(n_estimators=${p.n_estimators}, learning_rate=${p.learning_rate}, max_depth=${p.max_depth})),`);}
    if(b.kind==="adaboost"){imp.add("from sklearn.ensemble import AdaBoostClassifier");steps.push(`    ("model", AdaBoostClassifier(n_estimators=${p.n_estimators}, learning_rate=${p.learning_rate})),`);}
    if(b.kind==="logistic_regression"){imp.add("from sklearn.linear_model import LogisticRegression");steps.push(`    ("model", LogisticRegression(C=${p.C}, max_iter=${p.max_iter})),`);}
    if(b.kind==="svm"){imp.add("from sklearn.svm import SVC");steps.push(`    ("model", SVC(C=${p.C}, kernel=${Q(p.kernel)})),`);}
    if(b.kind==="knn"){imp.add("from sklearn.neighbors import KNeighborsClassifier");steps.push(`    ("model", KNeighborsClassifier(n_neighbors=${p.n_neighbors})),`);}
    if(b.kind==="naive_bayes"){imp.add("from sklearn.naive_bayes import GaussianNB");steps.push('    ("model", GaussianNB()),');}
    if(b.kind==="gradient_boosting"){imp.add("from sklearn.ensemble import GradientBoostingClassifier");steps.push(`    ("model", GradientBoostingClassifier(n_estimators=${p.n_estimators}, learning_rate=${p.learning_rate}, max_depth=${p.max_depth})),`);}
    if(b.kind==="extra_trees"){imp.add("from sklearn.ensemble import ExtraTreesClassifier");steps.push(`    ("model", ExtraTreesClassifier(n_estimators=${p.n_estimators}, max_depth=${p.max_depth})),`);}
    if(b.kind==="linear_regression"){imp.add("from sklearn.linear_model import LinearRegression");steps.push('    ("model", LinearRegression()),');}
    if(b.kind==="random_forest_regressor"){imp.add("from sklearn.ensemble import RandomForestRegressor");steps.push(`    ("model", RandomForestRegressor(n_estimators=${p.n_estimators}, max_depth=${p.max_depth})),`);}
    if(b.kind==="svr"){imp.add("from sklearn.svm import SVR");steps.push(`    ("model", SVR(C=${p.C}, kernel=${Q(p.kernel)})),`);}
    if(b.kind==="ridge"){imp.add("from sklearn.linear_model import Ridge");steps.push(`    ("model", Ridge(alpha=${p.alpha})),`);}
    if(b.kind==="lasso"){imp.add("from sklearn.linear_model import Lasso");steps.push(`    ("model", Lasso(alpha=${p.alpha})),`);}
    if(b.kind==="elastic_net"){imp.add("from sklearn.linear_model import ElasticNet");steps.push(`    ("model", ElasticNet(alpha=${p.alpha}, l1_ratio=${p.l1_ratio})),`);}
  });
  return`${[...imp].sort().join("\n")}
import pandas as pd

DATA_PATH = "data.csv"
TARGET = "label"
df = pd.read_csv(DATA_PATH)
X, y = df.drop(columns=[TARGET]), df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipeline = Pipeline([
${steps.join("\n")||'    ("model", RandomForestClassifier(n_estimators=300, random_state=42)),'}
])

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)
print(classification_report(y_test, preds))
`;
}
