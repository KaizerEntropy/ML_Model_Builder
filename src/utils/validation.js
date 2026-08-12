import {convLikeKinds,attentionKinds} from "../data/catalogs.js";
const normKinds=new Set(["batchnorm","groupnorm","instancenorm","layernorm","t_layernorm"]);
export function validateArchitecture(graph,shapes,mode){
  if(mode==="ml") return validateML(graph);
  if(mode==="transformer") return validateTransformer(graph,shapes);
  if(mode==="unet") return validateUnet(graph,shapes);
  return validateCNN(graph,shapes);
}
function validateCNN(graph,shapes){
  const issues=[];const p=(id,i,s,m)=>issues.push({blockId:id,index:i,severity:s,message:m});
  if(!graph.length) return{errors:[],warnings:[]};
  if(graph[0].kind!=="input") p(graph[0].id,0,"error","CNN must begin with an Input Tensor block.");
  const firstConv=graph.findIndex(b=>convLikeKinds.has(b.kind));
  let heads=0,lastHead=-1,hasFlat=false;
  graph.forEach((b,i)=>{
    const sh=shapes[i];
    if((firstConv<0||i<firstConv)&&b.kind!=="input"){
      if(b.kind==="activation") p(b.id,i,"warning","Activation before any conv layer is unusual.");
      if(normKinds.has(b.kind)) p(b.id,i,"warning","Normalization before any conv layer is unusual.");
      if(b.kind==="dropout"||b.kind==="dropout2d") p(b.id,i,"warning","Dropout before any conv layer is unusual.");
      if(attentionKinds.has(b.kind)) p(b.id,i,"warning","Attention before conv layers has no features.");
      if(["pool","avgpool"].includes(b.kind)) p(b.id,i,"warning","Pooling before conv has no spatial features.");
    }
    if(b.kind==="head"){heads++;lastHead=i;if(!hasFlat) p(b.id,i,"warning","Add Flatten or GAP before Classifier Head.");}
    if(["flatten","gap","gem","adapool"].includes(b.kind)){if(hasFlat) p(b.id,i,"warning","Duplicate spatial reduction.");hasFlat=true;}
    if(sh&&(sh.h<=0||sh.w<=0)) p(b.id,i,"error","Spatial dims collapsed to 0 — check kernel/stride/padding.");
  });
  if(heads>0&&lastHead!==graph.length-1) p(graph[lastHead].id,lastHead,"warning","Classifier Head should be last.");
  if(heads>1) p(null,-1,"warning","Multiple classifier heads detected.");
  return{errors:issues.filter(i=>i.severity==="error"),warnings:issues.filter(i=>i.severity==="warning")};
}
function validateTransformer(graph,shapes){
  const issues=[];const p=(id,i,s,m)=>issues.push({blockId:id,index:i,severity:s,message:m});
  if(!graph.length) return{errors:[],warnings:[]};
  const hasEmbed=graph.some(b=>["t_input","patch_embed"].includes(b.kind));
  if(!hasEmbed) p(graph[0].id,0,"error","Transformer needs an Input or Patch Embedding block first.");
  const hasPool=graph.some(b=>["t_cls_pool","t_mean_pool"].includes(b.kind));
  const hasHead=graph.some(b=>b.kind==="t_head");
  if(hasHead&&!hasPool) p(null,-1,"warning","Add CLS or Mean Pooling before Classification Head.");
  graph.forEach((b,i)=>{
    if(["mha","cross_attn","transformer_block","vit_block","swin_block","deit_block"].includes(b.kind)){
      const dim=Number(b.params.embed_dim||768),heads=Number(b.params.heads||8);
      if(dim%heads!==0) p(b.id,i,"error",`embed_dim (${dim}) must be divisible by heads (${heads}).`);
    }
  });
  return{errors:issues.filter(i=>i.severity==="error"),warnings:issues.filter(i=>i.severity==="warning")};
}
function validateUnet(graph,shapes){
  const issues=[];const p=(id,i,s,m)=>issues.push({blockId:id,index:i,severity:s,message:m});
  if(!graph.length) return{errors:[],warnings:[]};
  if(!["u_input","input"].includes(graph[0].kind)) p(graph[0].id,0,"error","U-Net must begin with an Input block.");
  const enc=graph.filter(b=>b.kind.startsWith("u_enc")).length;
  const dec=graph.filter(b=>b.kind.startsWith("u_dec")).length;
  if(enc>0&&dec===0) p(null,-1,"warning","U-Net needs decoder blocks to upsample.");
  if(dec>0&&enc===0) p(null,-1,"warning","U-Net needs encoder blocks before decoders.");
  const hasSeg=graph.some(b=>b.kind==="u_seg_head");
  if(!hasSeg) p(null,-1,"warning","Add a Segmentation Head for final output.");
  graph.forEach((b,i)=>{const sh=shapes[i];if(sh&&(sh.h<=0||sh.w<=0)) p(b.id,i,"error","Spatial dims collapsed.");});
  return{errors:issues.filter(i=>i.severity==="error"),warnings:issues.filter(i=>i.severity==="warning")};
}
function validateML(graph){
  const issues=[];const p=(id,i,s,m)=>issues.push({blockId:id,index:i,severity:s,message:m});
  if(!graph.length) return{errors:[],warnings:[]};
  const models=graph.filter(b=>!["tabular_input","image_features","standard_scaler","minmax_scaler","robust_scaler","pca","label_encoder","onehot_encoder","imputer"].includes(b.kind));
  if(models.length===0) p(null,-1,"warning","Pipeline has no model — add a classifier or regressor.");
  if(models.length>1) p(null,-1,"warning","Multiple models in pipeline — sklearn Pipeline supports one.");
  return{errors:issues.filter(i=>i.severity==="error"),warnings:issues.filter(i=>i.severity==="warning")};
}
export function issueMap(issues){
  const map=new Map();
  [...issues.errors,...issues.warnings].forEach(i=>{if(!i.blockId)return;const e=map.get(i.blockId);if(!e||(e==="warning"&&i.severity==="error"))map.set(i.blockId,i.severity);});
  return map;
}
