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
          <button className="icon-btn primary" onClick={() => setIsPlaying(!isPlaying)}>
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
  // A stylized abstract visualization of a tensor
  // We use CSS transforms to create a pseudo-3D block.
  // Dimensions are scaled logarithmically or clamped to keep UI manageable.
  
  const scale = (val) => Math.max(40, Math.min(200, Math.log2(val || 1) * 20));
  
  const width = scale(shape.w);
  const height = scale(shape.h);
  const depth = scale(shape.c) * 0.5; // Depth represents channels

  return (
    <div className="tensor-scene">
      <motion.div 
        className={`tensor-cube ${isInput ? 'input-cube' : ''}`}
        initial={{ rotateX: 60, rotateZ: -45, scale: 0.8 }}
        animate={{ rotateX: 60, rotateZ: -45, scale: 1 }}
        transition={{ type: "spring" }}
      >
        <div className="face top" style={{ width, height: depth, transform: `translateZ(${height}px)` }}></div>
        <div className="face front" style={{ width, height, transform: `translateY(${depth}px) rotateX(-90deg)` }}>
          <div className="grid-overlay"></div>
        </div>
        <div className="face right" style={{ width: depth, height, transform: `translateX(${width}px) translateY(${depth/2}px) rotateY(90deg) rotateX(-90deg)` }}></div>
      </motion.div>
      
      <div className="tensor-labels">
        <div className="label-c">C: {shape.c}</div>
        <div className="label-h">H: {shape.h}</div>
        <div className="label-w">W: {shape.w}</div>
      </div>
    </div>
  );
}
