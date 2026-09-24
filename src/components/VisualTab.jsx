import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, ChevronLeft, ChevronRight, Info } from "lucide-react";

const getBlockDescription = (block, shape) => {
  if (!block || !shape) return "No information available.";
  const p = block.params || {};
  switch (block.kind) {
    case "input":
    case "u_input":
    case "od_input":
    case "t_input":
    case "tabular_input":
      return `The initial data enters the network. The shape is (${shape.c_in} channels, ${shape.h_in} height, ${shape.w_in} width).`;
    case "conv":
    case "conv1x1":
    case "conv5x5":
    case "dilated":
    case "depthwise":
    case "grouped":
      return `A convolutional layer applies filters to extract spatial features. Using a kernel size of ${p.kernel}, it creates ${p.out} new feature maps. The spatial dimensions change to ${shape.h}x${shape.w} due to stride ${p.stride} and padding.`;
    case "resnet":
    case "bottleneck":
    case "resnext":
    case "mobilenet":
      return `A residual block. It applies convolutions and adds the original input (skip connection) to the output. This helps gradients flow through deep networks.`;
    case "pool":
    case "avgpool":
    case "adapool":
    case "gem":
    case "spp":
      return `A pooling layer reduces the spatial dimensions (from ${shape.h_in}x${shape.w_in} to ${shape.h}x${shape.w}) by aggregating local features, which provides translation invariance and reduces computational load.`;
    case "flatten":
    case "gap":
      return `Flattens the spatial dimensions into a 1D vector (or 1x1 spatial map). It aggregates the information across the entire image into ${shape.c} channels.`;
    case "head":
    case "t_head":
    case "yolo_head":
    case "ssd_head":
      return `The final classification or detection head. It maps the extracted high-dimensional features into the final ${p.classes} target classes/predictions.`;
    case "upsample":
    case "transpose_conv":
    case "pixelshuffle":
      return `Upsamples the spatial dimensions (from ${shape.h_in}x${shape.w_in} to ${shape.h}x${shape.w}), often used in decoders to reconstruct higher resolution details.`;
    case "patch_embed":
      return `Splits the image into patches (size ${p.patch_size}) and linearly embeds them into a sequence of tokens of dimension ${p.embed_dim}.`;
    case "transformer_block":
    case "mha":
      return `Self-attention mechanism allows the network to weigh the importance of different spatial patches or tokens relative to each other, capturing global context.`;
    default:
      if (block.kind.includes("enc")) return `Encoder block: extracts features and reduces spatial resolution.`;
      if (block.kind.includes("dec")) return `Decoder block: upsamples features and combines them with skip connections.`;
      return `Applies the ${block.label} operation to transform the feature maps.`;
  }
};

export default function VisualTab({ graph, shapes }) {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    let timer;
    if (isPlaying) {
      timer = setInterval(() => {
        setCurrentStep((prev) => (prev < graph.length - 1 ? prev + 1 : 0));
      }, 3000); // 3 seconds per step
    }
    return () => clearInterval(timer);
  }, [isPlaying, graph.length]);

  if (!graph || graph.length === 0) {
    return <div className="empty" style={{ margin: "40px auto" }}>No architecture built yet.</div>;
  }

  const block = graph[currentStep];
  const shape = shapes[currentStep];

  const goPrev = () => setCurrentStep((prev) => Math.max(0, prev - 1));
  const goNext = () => setCurrentStep((prev) => Math.min(graph.length - 1, prev + 1));

  return (
    <div className="visual-tab">
      <div className="visual-header">
        <h2>Interactive Model Execution</h2>
        <div className="controls">
          <button className="icon-btn" onClick={goPrev} disabled={currentStep === 0}><ChevronLeft size={18} /></button>
          <button className="icon-btn play-btn" onClick={() => setIsPlaying(!isPlaying)}>
            {isPlaying ? <Pause size={18} /> : <Play size={18} />}
          </button>
          <button className="icon-btn" onClick={goNext} disabled={currentStep === graph.length - 1}><ChevronRight size={18} /></button>
        </div>
      </div>

      <div className="visual-timeline">
        {graph.map((b, idx) => (
          <div key={b.id} className={`timeline-node ${idx === currentStep ? 'active' : ''} ${idx < currentStep ? 'past' : ''}`} onClick={() => setCurrentStep(idx)}>
            <div className="timeline-dot"></div>
            <span className="timeline-label">{b.label}</span>
          </div>
        ))}
      </div>

      <div className="visual-content">
        <AnimatePresence mode="wait">
          <motion.div
            key={currentStep}
            initial={{ opacity: 0, x: 40 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -40 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            className="visual-stage"
          >
            <div className="visual-tensor-container">
              {shape ? (
                <TensorVisualization shape={shape} isInput={currentStep === 0} blockKind={block.kind} />
              ) : (
                <div className="tensor-fallback">1D Data / Scalar</div>
              )}
            </div>

            <div className="visual-inspector">
              <div className="comment-box">
                <div className="comment-header"><Info size={16} /> <h3>What is happening here?</h3></div>
                <p>{getBlockDescription(block, shape)}</p>
              </div>

              <div className="block-details">
                <div className="detail-card">
                  <h4>Block Identity</h4>
                  <div className="detail-row"><span>Type:</span> <strong>{block.label}</strong></div>
                  <div className="detail-row"><span>ID:</span> <strong>{block.id.slice(0,5)}</strong></div>
                </div>

                <div className="detail-card">
                  <h4>Parameters</h4>
                  {Object.keys(block.params).length === 0 ? (
                    <div className="detail-row"><span style={{color: "var(--muted)"}}>No parameters</span></div>
                  ) : (
                    Object.entries(block.params).map(([k, v]) => (
                      <div className="detail-row" key={k}><span>{k}:</span> <strong>{String(v)}</strong></div>
                    ))
                  )}
                </div>

                {shape && (
                  <div className="detail-card shape-card">
                    <h4>Shape Transformation</h4>
                    <div className="shape-flow">
                      <div className="shape-badge in">[{shape.c_in}, {shape.h_in}, {shape.w_in}]</div>
                      <span className="shape-arrow">→</span>
                      <div className="shape-badge out">[{shape.c}, {shape.h}, {shape.w}]</div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}

function TensorVisualization({ shape, isInput, blockKind }) {
  const scale = (val) => Math.max(40, Math.min(220, Math.log2(val || 1) * 25));
  const width = scale(shape.w);
  const height = scale(shape.h);
  
  const layerCount = isInput ? 1 : (shape.c <= 3 ? shape.c : Math.min(5, Math.ceil(Math.log2(shape.c))));
  const spacing = 14; 
  const totalDepth = layerCount * spacing;

  return (
    <div className="tensor-scene">
      <motion.div 
        className={`tensor-cube ${isInput ? 'input-cube' : ''}`}
        initial={{ rotateX: 65, rotateZ: -45, scale: 0.8 }}
        animate={{ rotateX: 65, rotateZ: -45, scale: 1 }}
        transition={{ type: "spring" }}
        style={{ width, height, position: 'relative', transformStyle: 'preserve-3d' }}
      >
        {Array.from({ length: layerCount }).map((_, i) => (
          <div 
            key={i} 
            className="feature-plane" 
            style={{ 
              width, height, 
              position: 'absolute', 
              transform: `translateZ(${i * spacing}px)`,
              background: isInput ? 'rgba(var(--red-rgb), 0.15)' : 'rgba(var(--cyan-rgb), 0.15)',
              border: `1.5px solid rgba(${isInput ? 'var(--red-rgb)' : 'var(--cyan-rgb)'}, 0.6)`,
              boxShadow: `inset 0 0 16px rgba(${isInput ? 'var(--red-rgb)' : 'var(--cyan-rgb)'}, 0.2)`,
              borderRadius: '2px'
            }}
          >
            {(!isInput && i === layerCount - 1) && <div className="grid-overlay" style={{width: '100%', height: '100%', backgroundImage: 'linear-gradient(var(--line) 1px, transparent 1px), linear-gradient(90deg, var(--line) 1px, transparent 1px)', backgroundSize: '10px 10px'}} />}
          </div>
        ))}
        {(!isInput && blockKind && (blockKind.includes('conv') || blockKind.includes('pool'))) && (
           <motion.div 
              initial={{ x: 0, y: 0, opacity: 0 }}
              animate={{ x: width - 30, y: height - 30, opacity: 1 }}
              transition={{ duration: 2, repeat: Infinity, repeatType: "reverse", ease: "easeInOut" }}
              className="kernel-box" 
              style={{
                width: 30, height: 30, position: 'absolute',
                background: 'var(--panel2)',
                transform: `translateZ(${totalDepth + 4}px)`,
                boxShadow: '0 4px 16px rgba(0,0,0,0.6)',
                border: '2px solid var(--text)',
                borderRadius: '2px'
             }} 
           />
        )}
      </motion.div>
      
      <div className="tensor-labels" style={{ position: 'absolute', bottom: 20, display: 'flex', gap: 24, fontSize: 13, fontWeight: 'bold', color: isInput ? 'var(--red)' : 'var(--cyan)', fontFamily: 'JetBrains Mono, monospace' }}>
        <div className="label-c">C: {shape.c}</div>
        <div className="label-h">H: {shape.h}</div>
        <div className="label-w">W: {shape.w}</div>
      </div>
    </div>
  );
}
