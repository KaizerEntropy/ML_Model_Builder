export function convOut(size,kernel=3,stride=1,padding=1,dilation=1){
  return Math.max(1,Math.floor((size+2*Number(padding)-Number(dilation)*(Number(kernel)-1)-1)/Number(stride)+1));
}
export function calculateShapes(graph,dataset,mode){
  if(mode==="ml") return graph.map((b,i)=>({id:b.id,h:0,w:0,c:0,h_in:0,w_in:0,c_in:0,step:i+1}));
  if(mode==="transformer") return calcTransformerShapes(graph,dataset);
  if(mode==="unet") return calcUnetShapes(graph,dataset);
  let [h,w,c]=dataset.shape;
  return graph.map(block=>{
    const p=block.params,h_in=h,w_in=w,c_in=c;
    if(block.kind==="input"){h=Number(p.h||h);w=Number(p.w||w);c=Number(p.c||c)}
    else if(["conv","depthwise","grouped"].includes(block.kind)){h=convOut(h,p.kernel||3,p.stride||1,p.padding??1,p.dilation||1);w=convOut(w,p.kernel||3,p.stride||1,p.padding??1,p.dilation||1);c=Number(p.out||c)}
    else if(block.kind==="conv1x1"){c=Number(p.out||c)}
    else if(block.kind==="conv5x5"){h=convOut(h,5,p.stride||1,2,1);w=convOut(w,5,p.stride||1,2,1);c=Number(p.out||c)}
    else if(block.kind==="dilated"){h=convOut(h,p.kernel||3,1,Number(p.dilation||2),p.dilation||2);w=convOut(w,p.kernel||3,1,Number(p.dilation||2),p.dilation||2);c=Number(p.out||c)}
    else if(["bottleneck","inception","aspp","convnext","resnet","resnet_bottleneck","resnext","mobilenet","efficientnet","fusedmb","shuffle"].includes(block.kind)){h=Math.max(1,Math.floor(h/Number(p.stride||1)));w=Math.max(1,Math.floor(w/Number(p.stride||1)));c=Number(p.out||c)}
    else if(block.kind==="dense"){c+=Number(p.growth||16)*Number(p.layers||3)}
    else if(["pool","avgpool"].includes(block.kind)){h=Math.max(1,Math.floor(h/Number(p.stride||p.kernel||2)));w=Math.max(1,Math.floor(w/Number(p.stride||p.kernel||2)))}
    else if(block.kind==="adapool"){h=Number(p.h||1);w=Number(p.w||1)}
    else if(["gap","gem"].includes(block.kind)){h=1;w=1}
    else if(block.kind==="spp"){h=1;w=1;c*=String(p.bins||"1,2,4").split(",").reduce((s,n)=>s+Number(n)*Number(n),0)}
    else if(block.kind==="upsample"){const s=Number(p.scale||2);h*=s;w*=s}
    else if(block.kind==="transpose_conv"){const k=Number(p.kernel||4),s=Number(p.stride||2),pad=Number(p.padding||1);h=(h-1)*s-2*pad+k;w=(w-1)*s-2*pad+k;c=Number(p.out||c)}
    else if(block.kind==="pixelshuffle"){const s=Number(p.scale||2);h*=s;w*=s;c=Math.max(1,Math.floor(c/(s*s)))}
    else if(block.kind==="flatten"){c=h*w*c;h=1;w=1}
    else if(block.kind==="head"){c=Number(p.classes||10);h=1;w=1}
    return {id:block.id,h,w,c,h_in,w_in,c_in};
  });
}
function calcTransformerShapes(graph,dataset){
  let seq=197,dim=768;
  return graph.map(block=>{
    const p=block.params,s_in=seq,d_in=dim;
    if(block.kind==="t_input"){seq=Number(p.seq_len||197);dim=Number(p.embed_dim||768)}
    else if(block.kind==="patch_embed"){const ps=Number(p.patch_size||16),img=Number(p.img_size||224);seq=(img/ps)*(img/ps)+1;dim=Number(p.embed_dim||768)}
    else if(block.kind==="cls_token"){seq+=1}
    else if(block.kind==="t_head"){dim=Number(p.classes||10);seq=1}
    else if(block.kind==="t_cls_pool"||block.kind==="t_mean_pool"){seq=1}
    else if(["mha","cross_attn","masked_mha","transformer_block","decoder_block","vit_block","swin_block","deit_block","ffn"].includes(block.kind)){dim=Number(p.embed_dim||dim)}
    return {id:block.id,h:seq,w:1,c:dim,h_in:s_in,w_in:1,c_in:d_in,seq,dim};
  });
}
function calcUnetShapes(graph,dataset){
  let [h,w,c]=dataset.shape;
  return graph.map(block=>{
    const p=block.params,h_in=h,w_in=w,c_in=c;
    if(block.kind==="u_input"){h=Number(p.h||h);w=Number(p.w||w);c=Number(p.c||c)}
    else if(["u_enc","u_enc_res","u_enc_dense","u_enc_attn"].includes(block.kind)){c=Number(p.out||c);h=Math.max(1,Math.floor(h/2));w=Math.max(1,Math.floor(w/2))}
    else if(["u_bottleneck","u_aspp_bridge"].includes(block.kind)){c=Number(p.out||c)}
    else if(["u_dec","u_dec_res","u_dec_attn"].includes(block.kind)){c=Number(p.out||c);h*=2;w*=2}
    else if(block.kind==="u_seg_head"){c=Number(p.classes||2)}
    else if(["u_skip","u_attn_gate","u_deep_sup","unet_pp","resunet","transunet"].includes(block.kind)){c=Number(p.out||p.channels||c)}
    return {id:block.id,h,w,c,h_in,w_in,c_in};
  });
}
