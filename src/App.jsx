import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Bot, BrainCircuit, Box, GitBranch, Copy, Download, Image as ImageIcon, Plus, Search, Sparkles, Trash2, AlertTriangle, XCircle, ChevronLeft, ChevronRight, ChevronUp, ChevronDown, Target, FileCode } from "lucide-react";
import { toPng } from 'html-to-image';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { motion, AnimatePresence } from "framer-motion";

import { cnnCatalog, transformerCatalog, unetCatalog, mlCatalog, odCatalog, lossCatalog, activations } from "./data/catalogs.js";
import { imageDatasets, csvDatasets, augmentations, imageFilters } from "./data/datasets.js";
import { calculateShapes } from "./utils/shapeCalc.js";
import { validateArchitecture, issueMap } from "./utils/validation.js";
import { generateTorchCode, generateTransformerCode, generateUnetCode, generateSklearnCode, generateODCode } from "./utils/codeGen.js";
import VisualTab from "./components/VisualTab.jsx";

const uid = () => Math.random().toString(36).slice(2, 9);
function cloneBlock(template) { return { id: uid(), kind: template.kind, label: template.label, glyph: template.glyph, params: structuredClone(template.params || {}), attention: [] }; }

function makePreset(mode, dataset) {
  const cat = { cnn: cnnCatalog, transformer: transformerCatalog, unet: unetCatalog, ml: mlCatalog, od: odCatalog }[mode];
  const seq = { cnn: ["input", "conv", "resnet", "pool", "gap", "head"], transformer: ["t_input", "patch_embed", "pos_embed", "cls_token", "transformer_block", "t_cls_pool", "t_head"], unet: ["u_input", "u_enc", "u_enc", "u_bottleneck", "u_dec", "u_dec", "u_seg_head"], ml: ["tabular_input", "standard_scaler", "random_forest"], od: ["od_input", "cspdarknet", "panet", "yolo_head"] }[mode];
  const next = [];
  seq.forEach((kind) => {
    const template = cat.find(item => item.kind === kind);
    if (!template) return;
    const block = cloneBlock(template);
    if (block.kind === "input" || block.kind === "u_input" || block.kind === "od_input") block.params = { h: dataset.shape?.[0]||224, w: dataset.shape?.[1]||224, c: dataset.shape?.[2]||3 };
    if (["head", "t_head", "u_seg_head", "yolo_head", "yolox_head", "ssd_head", "retina_head", "faster_rcnn_head", "mask_rcnn_head", "centernet_head"].includes(block.kind)) block.params.classes = dataset.classes || 10;
    next.push(block);
  });
  return next;
}
function groupBy(items, key) { return items.reduce((acc, item) => { const group = item[key] || "Other"; acc[group] ||= []; acc[group].push(item); return acc; }, {}); }

export default function App() {
  const initialData = imageDatasets[0];
  const [mode, setMode] = useState("cnn");
  const [view, setView] = useState("builder");
  const [query, setQuery] = useState("");
  const [dataFilter, setDataFilter] = useState("All");
  
  const [leftOpen, setLeftOpen] = useState(true);
  const [rightOpen, setRightOpen] = useState(true);
  const [showCode, setShowCode] = useState(false);

  const [graphs, setGraphs] = useState({ cnn: makePreset("cnn", initialData), transformer: makePreset("transformer", initialData), unet: makePreset("unet", initialData), ml: makePreset("ml", csvDatasets[0]), od: makePreset("od", initialData) });
  const [losses, setLosses] = useState({ cnn: lossCatalog.cnn[0], transformer: lossCatalog.transformer[0], unet: lossCatalog.unet[0], ml: lossCatalog.ml[0], od: lossCatalog.od[0] });
  
  // Training Settings
  const [trainSettings, setTrainSettings] = useState({
    epochs: 10, batch_size: 32, lr: 0.001, optimizer: 'Adam', split: 0.2, kfold: 1, visualizations: ['plot_loss', 'confusion_matrix']
  });

  const [selectedId, setSelectedId] = useState(graphs.cnn[0]?.id || null);
  const [dataset, setDataset] = useState(initialData);
  const [selectedAugs, setSelectedAugs] = useState(["Resize", "HorizontalFlip", "Normalize"]);
  const [selectedFilter, setSelectedFilter] = useState("None");
  const [toast, setToast] = useState("");
  
  const [seqPaths, setSeqPaths] = useState([]);
  const [skipPaths, setSkipPaths] = useState([]);
  
  const [draggedId, setDraggedId] = useState(null);
  const [dropTargetIdx, setDropTargetIdx] = useState(null);
  const [dropErrors, setDropErrors] = useState([]);

  const canvasRef = useRef(null);
  const canvasInnerRef = useRef(null);
  const nodeRefs = useRef(new Map());
  const dragScrollRef = useRef(0);

  const graph = graphs[mode];
  const catalog = { cnn: cnnCatalog, transformer: transformerCatalog, unet: unetCatalog, ml: mlCatalog, od: odCatalog }[mode];
  
  const shapes = useMemo(() => calculateShapes(graph, dataset, mode), [graph, dataset, mode]);
  const validation = useMemo(() => validateArchitecture(graph, shapes, mode), [graph, shapes, mode]);
  const issues = useMemo(() => issueMap(validation), [validation]);
  
  const generatedCode = useMemo(() => {
    if (mode === "cnn") return generateTorchCode(graph, dataset, selectedAugs, losses.cnn, trainSettings, selectedFilter);
    if (mode === "transformer") return generateTransformerCode(graph, dataset, selectedAugs, losses.transformer, trainSettings, selectedFilter);
    if (mode === "unet") return generateUnetCode(graph, dataset, selectedAugs, losses.unet, trainSettings, selectedFilter);
    if (mode === "od") return generateODCode(graph, dataset, selectedAugs, losses.od, trainSettings, selectedFilter);
    return generateSklearnCode(graph, losses.ml, trainSettings);
  }, [mode, graph, dataset, selectedAugs, losses, trainSettings, selectedFilter]);

  const selected = graph.find(b => b.id === selectedId);

  useEffect(() => { if (toast) { const t = setTimeout(() => setToast(""), 2500); return () => clearTimeout(t); } }, [toast]);

  useLayoutEffect(() => {
    const renderPaths = () => {
      if (!canvasInnerRef.current) return;
      const box = canvasInnerRef.current.getBoundingClientRect();
      const sPaths = []; const kPaths = [];
      
      for (let i = 0; i < graph.length - 1; i++) {
        const from = nodeRefs.current.get(graph[i].id);
        const to = nodeRefs.current.get(graph[i+1].id);
        if (!from || !to) continue;
        const a = from.getBoundingClientRect();
        const b = to.getBoundingClientRect();
        
        let ax = a.left - box.left + a.width/2;
        let ay = a.top - box.top + a.height;
        let bx = b.left - box.left + b.width/2;
        let by = b.top - box.top;

        if (mode === "unet") {
           let startSide = 'bottom';
           let endSide = 'top';
           
           if (b.top < a.top - 20) {
               // B is visually ABOVE A (Going up the decoder)
               startSide = 'top';
               endSide = 'bottom';
           } else if (b.left > a.left + 50 && Math.abs(b.top - a.top) < 80) {
               // B is to the RIGHT of A (Bottleneck horizontal step)
               startSide = 'right';
               endSide = 'left';
           }
           
           if (startSide === 'top') {
               ay = a.top - box.top;
           } else if (startSide === 'right') {
               ax = a.right - box.left;
               ay = a.top - box.top + a.height/2;
           }
           
           if (endSide === 'bottom') {
               by = b.top - box.top + b.height + 4;
           } else if (endSide === 'left') {
               bx = b.left - box.left - 4;
               by = b.top - box.top + b.height/2;
           }
           
           const cp1x = startSide === 'right' ? ax + 40 : ax;
           const cp1y = startSide === 'bottom' ? ay + 40 : startSide === 'top' ? ay - 40 : ay;
           const cp2x = endSide === 'left' ? bx - 40 : bx;
           const cp2y = endSide === 'top' ? by - 40 : endSide === 'bottom' ? by + 40 : by;
           
           sPaths.push(`M ${ax} ${ay} C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${bx} ${by}`);
        } else {
           sPaths.push(`M ${ax} ${ay} L ${bx} ${by-4}`);
        }
      }
      
      if (mode === "unet") {
        const encs = graph.filter(b => b.kind.startsWith("u_enc"));
        const decs = graph.filter(b => b.kind.startsWith("u_dec"));
        const pairs = Math.min(encs.length, decs.length);
        for(let i = 0; i < pairs; i++) {
          const from = nodeRefs.current.get(encs[i].id);
          const to = nodeRefs.current.get(decs[decs.length - 1 - i].id);
          if (!from || !to) continue;
          const a = from.getBoundingClientRect();
          const bBox = to.getBoundingClientRect();
          const x1 = a.right - box.left;
          const y1 = a.top - box.top + a.height/2;
          const x2 = bBox.left - box.left;
          const y2 = bBox.top - box.top + bBox.height/2;
          kPaths.push({ d:`M ${x1} ${y1} L ${x2-4} ${y2}`, type: 'unet' });
        }
      }
      setSeqPaths(sPaths); setSkipPaths(kPaths);
    };
    const t = setTimeout(renderPaths, 50);
    window.addEventListener("resize", renderPaths);
    return () => { clearTimeout(t); window.removeEventListener("resize", renderPaths); };
  }, [graph, mode]);

  // Handle auto-scroll loop
  useEffect(() => {
    let animationFrame;
    const scrollLoop = () => {
      if (dragScrollRef.current !== 0 && canvasRef.current) {
        canvasRef.current.scrollTop += dragScrollRef.current;
      }
      animationFrame = requestAnimationFrame(scrollLoop);
    };
    if (draggedId) { animationFrame = requestAnimationFrame(scrollLoop); }
    else { dragScrollRef.current = 0; }
    return () => cancelAnimationFrame(animationFrame);
  }, [draggedId]);

  function handleDragOverCanvas(e) {
    e.preventDefault();
    if (!canvasRef.current || !draggedId) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const threshold = 60;
    if (e.clientY - rect.top < threshold) dragScrollRef.current = -8;
    else if (rect.bottom - e.clientY < threshold) dragScrollRef.current = 8;
    else dragScrollRef.current = 0;
  }

  function simulateValidation(nextGraph) {
    const s = calculateShapes(nextGraph, dataset, mode);
    const v = validateArchitecture(nextGraph, s, mode);
    return v.errors;
  }

  function handleDropTarget(e, idx) {
    e.preventDefault();
    if (dropTargetIdx !== idx) {
      setDropTargetIdx(idx);
      // Check validation
      const id = e.dataTransfer.getData("app/id");
      const kind = e.dataTransfer.getData("app/kind");
      let next = [...graph];
      if (id) {
        const from = graph.findIndex(b => b.id === id);
        if (from >= 0 && from !== idx) {
          const [b] = next.splice(from, 1);
          next.splice(from < idx ? idx - 1 : idx, 0, b);
          setDropErrors(simulateValidation(next));
        }
      } else if (kind) {
        const tmpl = catalog.find(i => i.kind === kind);
        if (tmpl) {
          next.splice(idx, 0, cloneBlock(tmpl));
          setDropErrors(simulateValidation(next));
        }
      }
    }
  }

  function handleDrop(e, targetIdx) {
    e.preventDefault();
    setDropTargetIdx(null);
    dragScrollRef.current = 0;
    
    if (dropErrors.length > 0) {
      setToast(dropErrors.map(e => e.message).join(" | "));
      return; // Reject drop!
    }

    const kind = e.dataTransfer.getData("app/kind");
    const id = e.dataTransfer.getData("app/id");
    if (id) {
      const from = graph.findIndex(b => b.id === id);
      if (from < 0 || from === targetIdx) return;
      const next = [...graph];
      const [b] = next.splice(from, 1);
      next.splice(from < targetIdx ? targetIdx - 1 : targetIdx, 0, b);
      setGraphs(p => ({ ...p, [mode]: next }));
      setSelectedId(id);
    } else if (kind) {
      const tmpl = catalog.find(i => i.kind === kind);
      if (tmpl) {
        const next = [...graph];
        const block = cloneBlock(tmpl);
        if (["input", "u_input", "od_input"].includes(block.kind)) block.params = { h: dataset.shape?.[0]||224, w: dataset.shape?.[1]||224, c: dataset.shape?.[2]||3 };
        if (["head", "t_head", "u_seg_head", "yolo_head", "yolox_head", "ssd_head", "retina_head", "faster_rcnn_head", "mask_rcnn_head", "centernet_head"].includes(block.kind)) block.params.classes = dataset.classes || 10;
        next.splice(targetIdx, 0, block);
        setGraphs(p => ({ ...p, [mode]: next }));
        setSelectedId(block.id);
      }
    }
  }

  function moveBlock(idx, direction) {
    if ((direction === -1 && idx === 0) || (direction === 1 && idx === graph.length - 1)) return;
    const next = [...graph];
    const [b] = next.splice(idx, 1);
    next.splice(idx + direction, 0, b);
    const errors = simulateValidation(next);
    if (errors.length > 0) {
      setToast(errors.map(e => e.message).join(" | "));
      return; // Reject move!
    }
    setGraphs(p => ({ ...p, [mode]: next }));
  }

  function captureDiagram() {
    if (!canvasInnerRef.current) return;
    setToast("Capturing diagram...");
    toPng(canvasInnerRef.current, { backgroundColor: '#050505' })
      .then(url => {
        const a = document.createElement("a");
        a.href = url; a.download = `model_forge_${mode}.png`; a.click();
        setToast("Diagram saved successfully!");
      });
  }

  let currentEnc = 1, currentDec = 1;
  const datasetsToRender = (mode === "ml" ? csvDatasets : imageDatasets)
    .filter(d => d.compatible.includes(mode) && (dataFilter === "All" || d.category.includes(dataFilter)));
  const catalogGroups = groupBy(catalog.filter(i => !query || `${i.group} ${i.label} ${i.kind}`.toLowerCase().includes(query.toLowerCase())), "group");

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="logo">MF</div>
          <div><h1>Model Forge</h1><p>Visual Architecture Builder • IDE Gen</p></div>
        </div>
        <nav className="tabs">
          {["builder", "visual", "data", "training", "code"].map(item => (
            <button key={item} className={view === item ? "active" : ""} onClick={() => setView(item)}>{item}</button>
          ))}
        </nav>
        <div className="actions">
          <button className="icon-btn" title="Copy Code" onClick={() => navigator.clipboard.writeText(generatedCode).then(()=>setToast("Code Copied!"))}><Copy size={18} /></button>
          <button className="icon-btn" title="Download Code" onClick={() => {
            const blob = new Blob([generatedCode], { type: "text/plain" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a"); a.href = url; a.download = `model_${mode}.py`; a.click();
          }}><Download size={18} /></button>
          <button className="primary" onClick={() => window.open("https://kaggle.com/code/new", "_blank")}>Run in Kaggle</button>
        </div>
      </header>

      <main className={`workspace ${leftOpen?"":"close-left"} ${rightOpen?"":"close-right"}`}>
        {leftOpen && (
          <aside className="panel">
            <div className="panel-header">
              <div><span>Library</span><strong>{mode.toUpperCase()} Blocks</strong></div>
              <div className="actions">
                <button className="text-btn" onClick={() => setGraphs(p=>({...p,[mode]:makePreset(mode, dataset)}))}><Sparkles size={14} /> Preset</button>
                <button className="icon-btn mini" onClick={() => setLeftOpen(false)}><ChevronLeft size={14}/></button>
              </div>
            </div>
            <div className="mode-switch">
              <button className={mode === "cnn" ? "active" : ""} onClick={() => {setMode("cnn"); setDataset(imageDatasets[0]); setDataFilter("All");}}><BrainCircuit size={14}/> CNN</button>
              <button className={mode === "transformer" ? "active" : ""} onClick={() => {setMode("transformer"); setDataset(imageDatasets[0]); setDataFilter("All");}}><Box size={14}/> ViT</button>
              <button className={mode === "unet" ? "active" : ""} onClick={() => {setMode("unet"); setDataset(imageDatasets[0]); setDataFilter("All");}}><GitBranch size={14}/> U-Net</button>
              <button className={mode === "od" ? "active" : ""} onClick={() => {setMode("od"); setDataset(imageDatasets.find(d=>d.compatible==="od") || imageDatasets[0]); setDataFilter("All");}}><Target size={14}/> OD</button>
              <button className={mode === "ml" ? "active" : ""} onClick={() => {setMode("ml"); setDataset(csvDatasets[0]); setDataFilter("All");}}><Bot size={14}/> ML</button>
            </div>
            <label className="search-row"><Search size={16}/><input value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search library..."/></label>
            <div className="scroll">
              {Object.entries(catalogGroups).map(([group, items]) => (
                <section className="section" key={group}>
                  <h2>{group}</h2>
                  <div className="palette-list">
                    {items.map(item => (
                      <div key={item.kind} className={`palette-item ${mode}`} draggable onDragStart={e => e.dataTransfer.setData("app/kind", item.kind)}>
                        <span className="glyph">{item.glyph}</span>
                        <div><strong>{item.label}</strong><small>{item.meta}</small></div>
                        <Plus size={16} />
                      </div>
                    ))}
                  </div>
                </section>
              ))}
            </div>
          </aside>
        )}
        {!leftOpen && <div className="collapsed-bar" onClick={()=>setLeftOpen(true)}><ChevronRight size={18}/></div>}

        {view === "builder" && (
          <section className="panel canvas-panel">
            <div className="canvas-toolbar">
              <div className="chip-row">
                <span className="chip hot">{dataset.name}</span>
                <span className="chip shape">{graph.length} blocks</span>
              </div>
              <div className="toolbar-actions">
                <button onClick={() => setShowCode(!showCode)}><FileCode size={16}/> {showCode ? "Hide Live Code" : "Show Live Code"}</button>
                <button onClick={captureDiagram}><ImageIcon size={16}/> Export Diagram</button>
                <button onClick={() => setGraphs(p=>({...p,[mode]:[]}))}><Trash2 size={16}/> Clear Canvas</button>
              </div>
            </div>

            <div className="canvas-container" ref={canvasRef} onDragOver={handleDragOverCanvas} onDrop={e=>handleDrop(e, graph.length)} style={{ display: 'flex' }}>
              <div className="canvas-inner" ref={canvasInnerRef} style={{ flex: 1, minWidth: 0 }}>
                <svg className="svg-layer" aria-hidden="true">
                  <defs>
                    <marker id="arrowhead" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto"><polygon points="0 0, 8 4, 0 8" fill="var(--red)" /></marker>
                    <marker id="arrowhead-unet" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto"><polygon points="0 0, 8 4, 0 8" fill="var(--teal)" /></marker>
                  </defs>
                  {seqPaths.map((p, i) => <path key={'s'+i} className="seq-arrow" d={p} markerEnd="url(#arrowhead)" />)}
                  {skipPaths.map((p, i) => <path key={'k'+i} className="skip-path-unet" d={p.d} markerEnd="url(#arrowhead-unet)" />)}
                </svg>
                <div className={`pipeline ${mode === 'unet' ? 'unet-layout' : ''}`}>
                  {!graph.length && <div className="empty">Drag & drop architecture blocks here</div>}
                <AnimatePresence>
                  {graph.map((block, idx) => {
                    const gridStyle = {};
                    if (mode === "unet") {
                      if (block.kind.startsWith("u_enc") || block.kind === "u_input") { gridStyle.gridColumn = 1; gridStyle.gridRow = currentEnc++; }
                      else if (block.kind.includes("bottleneck") || block.kind.includes("bridge")) { gridStyle.gridColumn = 2; gridStyle.gridRow = Math.max(currentEnc, 2); }
                      else { gridStyle.gridColumn = 3; gridStyle.gridRow = Math.max(1, currentEnc - currentDec); currentDec++; }
                    }
                    return (
                      <motion.div 
                        layout 
                        initial={{ opacity: 0, scale: 0.8 }} 
                        animate={{ opacity: 1, scale: 1 }} 
                        exit={{ opacity: 0, scale: 0.8, transition: { duration: 0.15 } }}
                        key={block.id} 
                        className="node-wrapper" 
                        style={gridStyle}
                      >
                        {dropTargetIdx === idx && <div className={`drop-indicator ${dropErrors.length>0?'invalid':''}`}/>}
                        <article 
                          ref={el => el ? nodeRefs.current.set(block.id, el) : nodeRefs.current.delete(block.id)}
                          className={`node ${mode} ${selectedId === block.id ? "selected" : ""} ${issues.get(block.id)==="error"?"v-error":issues.get(block.id)==="warning"?"v-warning":""} ${draggedId===block.id?"dragging":""}`}
                          onClick={() => setSelectedId(block.id)}
                          draggable
                          onDragStart={e => { e.dataTransfer.setData("app/id", block.id); setDraggedId(block.id); }}
                          onDragEnd={() => { setDraggedId(null); setDropTargetIdx(null); setDropErrors([]); dragScrollRef.current=0; }}
                          onDragOver={e => handleDropTarget(e, idx)}
                          onDrop={e => { e.stopPropagation(); handleDrop(e, idx); }}
                        >
                          <div className="node-top">
                            <div className="drag-handle"><span/><span/><span/></div>
                            <span className="glyph">{block.glyph}</span>
                            <div><strong>{block.label}</strong><small>{block.kind}</small></div>
                            <div style={{ display: 'flex', gap: 2 }}>
                              {mode !== "unet" && (
                                <>
                                  <button className="icon-btn mini" onClick={(e) => { e.stopPropagation(); moveBlock(idx, -1); }} disabled={idx === 0}><ChevronUp size={14}/></button>
                                  <button className="icon-btn mini" onClick={(e) => { e.stopPropagation(); moveBlock(idx, 1); }} disabled={idx === graph.length - 1}><ChevronDown size={14}/></button>
                                </>
                              )}
                              <button className="icon-btn mini" onClick={e=>{e.stopPropagation(); setGraphs(p=>({...p,[mode]:graph.filter(b=>b.id!==block.id)}))}}><Trash2 size={14}/></button>
                            </div>
                          </div>
                          <div className="node-body">
                            {mode !== "ml" && shapes[idx] && <span className="chip dim-flow">{shapes[idx].h_in}×{shapes[idx].w_in}×{shapes[idx].c_in} → {shapes[idx].h}×{shapes[idx].w}×{shapes[idx].c}</span>}
                          </div>
                        </article>
                      </motion.div>
                    );
                  })}
                </AnimatePresence>
                {dropTargetIdx === graph.length && <div className={`drop-indicator ${dropErrors.length>0?'invalid':''}`} style={{bottom: '-20px', top: 'auto'}}/>}
                </div>
              </div>
              
              {showCode && (
                <div className="live-code-panel" style={{ width: '450px', borderLeft: '1px solid var(--line-soft)', backgroundColor: '#1e1e1e', overflowY: 'auto', flexShrink: 0 }}>
                  <div style={{ padding: '8px 12px', borderBottom: '1px solid var(--line)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', position: 'sticky', top: 0, backgroundColor: '#1e1e1e', zIndex: 10 }}>
                    <strong style={{ color: '#fff', fontSize: '12px' }}>Live Code Preview</strong>
                    <button className="icon-btn mini" style={{ minHeight: '24px', width: '24px' }} onClick={() => setShowCode(false)}><XCircle size={14}/></button>
                  </div>
                  <SyntaxHighlighter language="python" style={vscDarkPlus} customStyle={{ margin: 0, padding: '16px', fontSize: '11.5px', background: 'transparent' }}>
                    {generatedCode}
                  </SyntaxHighlighter>
                </div>
              )}
            </div>
          </section>
        )}

        {view !== "builder" && (
          <section className="panel" style={{gridColumn: "2 / span 2"}}>
            {view === "data" && (
              <div className="scroll">
                <h2>Compatible Datasets</h2>
                <div className="tabs" style={{marginBottom:16, overflowX:'auto', whiteSpace:'nowrap'}}>
                  {["All", ...new Set((mode === "ml" ? csvDatasets : imageDatasets).filter(d => d.compatible.includes(mode)).map(d => d.category))].map(t => (
                    <button key={t} className={dataFilter === t ? "active" : ""} onClick={()=>setDataFilter(t)}>{t}</button>
                  ))}
                </div>
                <div className="dataset-grid">
                  {datasetsToRender.map(d => (
                    <div key={d.name} className={`dataset-card ${dataset.name === d.name ? "active" : ""}`} onClick={() => setDataset(d)}>
                      <div><strong>{d.name}</strong> <a href={d.url} target="_blank" rel="noreferrer">Source</a></div>
                      <small>{d.category} • {d.classes||d.target} • {d.size||d.samples}</small>
                    </div>
                  ))}
                </div>
                {mode !== "ml" && (
                  <div style={{marginTop: 24}}>
                    <h2>Image Augmentations</h2>
                    <div className="check-grid">
                      {augmentations.map(a => (
                        <label key={a} className="check"><input type="checkbox" checked={selectedAugs.includes(a)} onChange={() => setSelectedAugs(p => p.includes(a)?p.filter(x=>x!==a):[...p,a])}/> {a}</label>
                      ))}
                    </div>
                    <h2 style={{marginTop: 24}}>Image Filters (Only one selectable)</h2>
                    <div className="check-grid">
                      {imageFilters.map(f => (
                        <label key={f} className="check"><input type="radio" name="imageFilter" checked={selectedFilter === f} onChange={() => setSelectedFilter(f)}/> {f}</label>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
            
            {view === "training" && (
              <div className="scroll">
                <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:20}}>
                  <div>
                    <h2>Training Loop Hyperparameters</h2>
                    <div className="form-grid">
                      <label>Epochs <input type="number" value={trainSettings.epochs} onChange={e=>setTrainSettings(p=>({...p, epochs:Number(e.target.value)}))}/></label>
                      <label>Batch Size <input type="number" value={trainSettings.batch_size} onChange={e=>setTrainSettings(p=>({...p, batch_size:Number(e.target.value)}))}/></label>
                      <label>Learning Rate <input type="number" step="0.0001" value={trainSettings.lr} onChange={e=>setTrainSettings(p=>({...p, lr:Number(e.target.value)}))}/></label>
                      <label>Optimizer 
                        <select value={trainSettings.optimizer} onChange={e=>setTrainSettings(p=>({...p, optimizer:e.target.value}))}>
                          <option>Adam</option><option>SGD</option><option>RMSprop</option><option>AdamW</option>
                        </select>
                      </label>
                      <label>Train/Val Split (%) <input type="number" step="0.05" value={trainSettings.split} onChange={e=>setTrainSettings(p=>({...p, split:Number(e.target.value)}))}/></label>
                      <label>k-Fold Splits <input type="number" value={trainSettings.kfold} onChange={e=>setTrainSettings(p=>({...p, kfold:Number(e.target.value)}))}/></label>
                    </div>
                  </div>
                  <div>
                    <h2>Visualizations & Tracking</h2>
                    <div className="form-grid" style={{gridTemplateColumns:'1fr'}}>
                      {(mode === "unet" ? ["plot_loss", "mask_overlay", "iou_trend", "prediction_samples"] : mode === "od" ? ["plot_loss", "bbox_overlay", "map_trend"] : ["plot_loss", "confusion_matrix", "tsne", "gradcam"]).map(v => (
                         <label key={v} className="check"><input type="checkbox" checked={trainSettings.visualizations.includes(v)} onChange={()=>setTrainSettings(p=>({...p, visualizations: p.visualizations.includes(v)?p.visualizations.filter(x=>x!==v):[...p.visualizations, v]}))} /> Generate {v.replace(/_/g," ")} Code</label>
                      ))}
                    </div>
                  </div>
                </div>

                <div style={{marginTop: 32}}>
                  <h2>Loss Function Selection</h2>
                  <div className="loss-grid">
                    {lossCatalog[mode].map(l => (
                      <div key={l.kind} className={`loss-card ${losses[mode].kind === l.kind ? "active" : ""}`} onClick={() => setLosses(p => ({...p, [mode]: l}))}>
                        <div><strong>{l.label}</strong><small>{l.meta}</small></div>
                      </div>
                    ))}
                  </div>
                  {losses[mode] && Object.keys(losses[mode].params).length > 0 && (
                    <div style={{marginTop: 20}}>
                      <h2>Loss Hyperparameters</h2>
                      <div className="form-grid">
                        {Object.entries(losses[mode].params).map(([k,v]) => (
                          <label key={k}>{k} <input type="number" step="0.1" value={v} onChange={e => setLosses(p => ({...p, [mode]: {...p[mode], params:{...p[mode].params, [k]: Number(e.target.value)}}}) )}/></label>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
            
            {view === "code" && (
               <div style={{height:'100%', overflow:'auto', backgroundColor:'#1e1e1e', margin:-12}}>
                  <SyntaxHighlighter language="python" style={vscDarkPlus} showLineNumbers customStyle={{margin:0, padding:'24px', fontSize:'13px', background:'transparent'}}>
                    {generatedCode}
                  </SyntaxHighlighter>
               </div>
            )}
            
            {view === "visual" && (
              <VisualTab graph={graph} shapes={shapes} />
            )}
          </section>
        )}

        {view === "builder" && (
          rightOpen ? (
            <aside className="panel inspector-panel">
              <div className="panel-header">
                <button className="icon-btn mini" onClick={() => setRightOpen(false)}><ChevronRight size={14}/></button>
                <div><span>Inspector</span><strong>Block Settings</strong></div>
              </div>
              <div className="inspector-scroll scroll">
                {!selected ? <div className="empty" style={{marginTop:40, minHeight:80}}>Select a block to inspect</div> : (
                  <>
                    <div className="section">
                      <h2>{selected.label} Config</h2>
                      <div className="form-grid">
                        {Object.keys(selected.params).map(k => (
                          k !== "skipTo" && (
                            <label key={k}>{k.replace("_", " ")}
                              {k === "activation" || k === "mode" ? (
                                <select value={selected.params[k]} onChange={e => setGraphs(p=>({...p, [mode]: graph.map(b=>b.id===selected.id?{...b,params:{...b.params,[k]:e.target.value}}:b)}))}>
                                  {(k==="activation"?activations:["nearest","bilinear"]).map(o=><option key={o} value={o}>{o}</option>)}
                                </select>
                              ) : (
                                <input type={typeof selected.params[k]==="number"?"number":"text"} value={selected.params[k]} onChange={e => setGraphs(p=>({...p, [mode]: graph.map(b=>b.id===selected.id?{...b,params:{...b.params,[k]:typeof selected.params[k]==="number"?Number(e.target.value):e.target.value}}:b)}))}/>
                              )}
                            </label>
                          )
                        ))}
                      </div>
                    </div>
                  </>
                )}
              </div>
            </aside>
          ) : (
            <div className="collapsed-bar right" onClick={()=>setRightOpen(true)}><ChevronLeft size={18}/></div>
          )
        )}
      </main>
      <div className={`toast ${toast?"show":""}`}>{toast}</div>
    </div>
  );
}
