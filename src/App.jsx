import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Bot, BrainCircuit, Box, GitBranch, Copy, Download, Image as ImageIcon, Plus, Search, Sparkles, Trash2, AlertTriangle, XCircle } from "lucide-react";
import { toPng } from 'html-to-image';

import { cnnCatalog, transformerCatalog, unetCatalog, mlCatalog, lossCatalog, attentionKinds, residualKinds, convLikeKinds, activations } from "./data/catalogs.js";
import { imageDatasets, csvDatasets, augmentations } from "./data/datasets.js";
import { calculateShapes } from "./utils/shapeCalc.js";
import { validateArchitecture, issueMap } from "./utils/validation.js";
import { generateTorchCode, generateTransformerCode, generateUnetCode, generateSklearnCode } from "./utils/codeGen.js";

const uid = () => Math.random().toString(36).slice(2, 9);

function cloneBlock(template) {
  return { id: uid(), kind: template.kind, label: template.label, glyph: template.glyph, params: structuredClone(template.params || {}), attention: [] };
}

function makePreset(mode, dataset) {
  const cat = { cnn: cnnCatalog, transformer: transformerCatalog, unet: unetCatalog, ml: mlCatalog }[mode];
  const seq = {
    cnn: ["input", "conv", "resnet", "pool", "gap", "head"],
    transformer: ["t_input", "patch_embed", "pos_embed", "cls_token", "transformer_block", "t_cls_pool", "t_head"],
    unet: ["u_input", "u_enc", "u_enc", "u_bottleneck", "u_dec", "u_dec", "u_seg_head"],
    ml: ["tabular_input", "standard_scaler", "random_forest"]
  }[mode];
  
  const next = [];
  seq.forEach((kind) => {
    const template = cat.find(item => item.kind === kind);
    if (!template) return;
    const block = cloneBlock(template);
    if (block.kind === "input" || block.kind === "u_input") {
      const [h, w, c] = dataset.shape || [224,224,3];
      block.params = { h, w, c };
    }
    if (block.kind === "head" || block.kind === "t_head" || block.kind === "u_seg_head") block.params.classes = dataset.classes || 10;
    next.push(block);
  });
  return next;
}

function groupBy(items, key) {
  return items.reduce((acc, item) => {
    const group = item[key] || "Other";
    acc[group] ||= [];
    acc[group].push(item);
    return acc;
  }, {});
}

function downloadFile(name, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click();
  URL.revokeObjectURL(url);
}

export default function App() {
  const initialData = imageDatasets[0];
  const [mode, setMode] = useState("cnn");
  const [view, setView] = useState("builder");
  const [query, setQuery] = useState("");
  
  const [graphs, setGraphs] = useState({
    cnn: makePreset("cnn", initialData),
    transformer: makePreset("transformer", initialData),
    unet: makePreset("unet", initialData),
    ml: makePreset("ml", csvDatasets[0]),
  });
  
  const [losses, setLosses] = useState({
    cnn: lossCatalog.cnn[0], transformer: lossCatalog.transformer[0], unet: lossCatalog.unet[0], ml: lossCatalog.ml[0]
  });
  
  const [selectedId, setSelectedId] = useState(graphs.cnn[0]?.id || null);
  const [dataset, setDataset] = useState(initialData);
  const [selectedAugs, setSelectedAugs] = useState(["Resize", "HorizontalFlip", "Normalize"]);
  const [toast, setToast] = useState("");
  
  const [seqPaths, setSeqPaths] = useState([]);
  const [skipPaths, setSkipPaths] = useState([]);
  
  const [draggedId, setDraggedId] = useState(null);
  const [dropTargetIdx, setDropTargetIdx] = useState(null);

  const canvasRef = useRef(null);
  const canvasInnerRef = useRef(null);
  const nodeRefs = useRef(new Map());

  const graph = graphs[mode];
  const catalog = { cnn: cnnCatalog, transformer: transformerCatalog, unet: unetCatalog, ml: mlCatalog }[mode];
  
  const shapes = useMemo(() => calculateShapes(graph, dataset, mode), [graph, dataset, mode]);
  const validation = useMemo(() => validateArchitecture(graph, shapes, mode), [graph, shapes, mode]);
  const issues = useMemo(() => issueMap(validation), [validation]);
  
  const generatedCode = useMemo(() => {
    if (mode === "cnn") return generateTorchCode(graph, dataset, selectedAugs, losses.cnn);
    if (mode === "transformer") return generateTransformerCode(graph, dataset, selectedAugs, losses.transformer);
    if (mode === "unet") return generateUnetCode(graph, dataset, selectedAugs, losses.unet);
    return generateSklearnCode(graph, losses.ml);
  }, [mode, graph, dataset, selectedAugs, losses]);

  const selected = graph.find(b => b.id === selectedId);

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(""), 2500);
    return () => clearTimeout(t);
  }, [toast]);

  useLayoutEffect(() => {
    const renderPaths = () => {
      if (!canvasInnerRef.current) return;
      const box = canvasInnerRef.current.getBoundingClientRect();
      const sPaths = [];
      const kPaths = [];
      
      // 1. Sequence Arrows
      for (let i = 0; i < graph.length - 1; i++) {
        const from = nodeRefs.current.get(graph[i].id);
        const to = nodeRefs.current.get(graph[i+1].id);
        if (!from || !to) continue;
        const a = from.getBoundingClientRect();
        const b = to.getBoundingClientRect();
        const ax = a.left - box.left + a.width/2;
        const ay = a.top - box.top + a.height;
        const bx = b.left - box.left + b.width/2;
        const by = b.top - box.top;
        
        if (mode === "unet") {
           // Curve for U-Net layout transitions
           sPaths.push(`M ${ax} ${ay} C ${ax} ${ay+40}, ${bx} ${by-40}, ${bx} ${by-4}`);
        } else {
           sPaths.push(`M ${ax} ${ay} L ${bx} ${by-4}`);
        }
      }
      
      // 2. ResNet Skip Paths
      graph.forEach(b => {
        if (!b.params.skipTo) return;
        const from = nodeRefs.current.get(b.id);
        const to = nodeRefs.current.get(b.params.skipTo);
        if (!from || !to) return;
        const a = from.getBoundingClientRect();
        const bBox = to.getBoundingClientRect();
        const x1 = a.right - box.left + 8;
        const y1 = a.top - box.top + a.height * 0.3;
        const x2 = bBox.right - box.left + 8;
        const y2 = bBox.top - box.top + bBox.height * 0.7;
        const rail = Math.max(x1, x2) + 60;
        kPaths.push({ d:`M ${x1} ${y1} C ${rail} ${y1}, ${rail} ${y2}, ${x2+4} ${y2}`, type: 'resnet' });
      });

      // 3. U-Net Skip Paths (Auto-generated from Enc -> Dec)
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
      
      setSeqPaths(sPaths);
      setSkipPaths(kPaths);
    };

    // Small timeout ensures DOM has fully painted the new grid layout before we calculate boxes
    const t = setTimeout(renderPaths, 50);
    window.addEventListener("resize", renderPaths);
    return () => { clearTimeout(t); window.removeEventListener("resize", renderPaths); };
  }, [graph, mode]);

  function updateGraph(nextGraph) {
    setGraphs(p => ({ ...p, [mode]: nextGraph }));
  }

  function addBlock(kind, idx = graph.length) {
    const template = catalog.find(i => i.kind === kind);
    if (!template) return;
    const next = [...graph];
    const block = cloneBlock(template);
    if (["input", "u_input"].includes(block.kind)) {
      block.params = { h: dataset.shape?.[0]||224, w: dataset.shape?.[1]||224, c: dataset.shape?.[2]||3 };
    }
    if (["head", "t_head", "u_seg_head"].includes(block.kind)) block.params.classes = dataset.classes || 10;
    next.splice(idx, 0, block);
    updateGraph(next);
    setSelectedId(block.id);
  }

  function handleDropTarget(e, idx) {
    e.preventDefault();
    setDropTargetIdx(idx);
  }
  
  function handleDrop(e, targetIdx) {
    e.preventDefault();
    setDropTargetIdx(null);
    const kind = e.dataTransfer.getData("app/kind");
    const id = e.dataTransfer.getData("app/id");
    
    if (id) {
      const from = graph.findIndex(b => b.id === id);
      if (from < 0 || from === targetIdx) return;
      const next = [...graph];
      const [b] = next.splice(from, 1);
      next.splice(from < targetIdx ? targetIdx - 1 : targetIdx, 0, b);
      updateGraph(next);
      setSelectedId(id);
    } else if (kind) {
      addBlock(kind, targetIdx);
    }
  }

  function captureDiagram() {
    if (!canvasInnerRef.current) return;
    setToast("Capturing diagram...");
    toPng(canvasInnerRef.current, { backgroundColor: '#060608', style: { transform: 'scale(1)', transformOrigin: 'top left' }})
      .then(url => {
        const a = document.createElement("a");
        a.href = url; a.download = `model_forge_${mode}.png`; a.click();
        setToast("Diagram saved successfully!");
      })
      .catch(() => setToast("Failed to capture diagram"));
  }

  const catalogGroups = groupBy(catalog.filter(i => !query || `${i.group} ${i.label} ${i.kind}`.toLowerCase().includes(query.toLowerCase())), "group");

  // Calculate U-Net grid positions
  let currentEnc = 1, currentDec = 1;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="logo">MF</div>
          <div>
            <h1>Model Forge</h1>
            <p>Visual Architecture Builder • PyTorch & Scikit-Learn</p>
          </div>
        </div>
        <nav className="tabs">
          {["builder", "data", "loss", "code"].map(item => (
            <button key={item} className={view === item ? "active" : ""} onClick={() => setView(item)}>{item}</button>
          ))}
        </nav>
        <div className="actions">
          <button className="icon-btn" title="Copy Code" onClick={() => navigator.clipboard.writeText(generatedCode).then(()=>setToast("Code Copied!"))}><Copy size={18} /></button>
          <button className="icon-btn" title="Download Code" onClick={() => downloadFile(`model_${mode}.py`, generatedCode, "text/plain")}><Download size={18} /></button>
          <button className="primary" onClick={() => window.open("https://kaggle.com/code/new", "_blank")}>Run in Kaggle</button>
        </div>
      </header>

      <main className="workspace">
        <aside className="panel">
          <div className="panel-header">
            <div><span>Library</span><strong>{mode.toUpperCase()} Blocks</strong></div>
            <button onClick={() => { updateGraph(makePreset(mode, dataset)); setSelectedId(null); }}><Sparkles size={14} /> Preset</button>
          </div>
          <div className="mode-switch">
            <button className={mode === "cnn" ? "active" : ""} onClick={() => {setMode("cnn"); setDataset(imageDatasets[0]);}}><BrainCircuit size={14}/> CNN</button>
            <button className={mode === "transformer" ? "active" : ""} onClick={() => {setMode("transformer"); setDataset(imageDatasets[0]);}}><Box size={14}/> ViT</button>
            <button className={mode === "unet" ? "active" : ""} onClick={() => {setMode("unet"); setDataset(imageDatasets[0]);}}><GitBranch size={14}/> U-Net</button>
            <button className={mode === "ml" ? "active" : ""} onClick={() => {setMode("ml"); setDataset(csvDatasets[0]);}}><Bot size={14}/> ML</button>
          </div>
          <label className="search-row"><Search size={16}/><input value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search library..."/></label>
          <div className="scroll">
            {Object.entries(catalogGroups).map(([group, items]) => (
              <section className="section" key={group}>
                <h2>{group}</h2>
                <div className="palette-list">
                  {items.map(item => (
                    <div key={item.kind} className={`palette-item ${mode}`} draggable onDragStart={e => e.dataTransfer.setData("app/kind", item.kind)} onClick={() => addBlock(item.kind)}>
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

        {view === "builder" && (
          <section className="panel canvas-panel">
            <div className="canvas-toolbar">
              <div className="chip-row">
                <span className="chip hot">{dataset.name}</span>
                <span className="chip shape">{graph.length} blocks</span>
              </div>
              <div className="toolbar-actions">
                <button onClick={captureDiagram}><ImageIcon size={16}/> Export Diagram</button>
                <button onClick={() => updateGraph([])}><Trash2 size={16}/> Clear Canvas</button>
              </div>
            </div>
            
            {(validation.errors.length > 0 || validation.warnings.length > 0) && (
              <div className="validation-banner">
                {validation.errors.map((e, i) => <div key={'e'+i} className="v-error"><XCircle size={14} className="v-icon"/> {e.message}</div>)}
                {validation.warnings.map((w, i) => <div key={'w'+i} className="v-warn"><AlertTriangle size={14} className="v-icon"/> {w.message}</div>)}
              </div>
            )}

            <div className="canvas-container" ref={canvasRef} onDragOver={e=>e.preventDefault()} onDrop={e=>handleDrop(e, graph.length)}>
              <div className="canvas-inner" ref={canvasInnerRef}>
                <svg className="svg-layer" aria-hidden="true">
                  <defs>
                    <marker id="arrowhead" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto">
                      <polygon points="0 0, 8 4, 0 8" fill="var(--red)" />
                    </marker>
                    <marker id="arrowhead-resnet" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto">
                      <polygon points="0 0, 8 4, 0 8" fill="var(--amber)" />
                    </marker>
                    <marker id="arrowhead-unet" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto">
                      <polygon points="0 0, 8 4, 0 8" fill="var(--teal)" />
                    </marker>
                  </defs>
                  {seqPaths.map((p, i) => <path key={'s'+i} className="seq-arrow" d={p} markerEnd="url(#arrowhead)" />)}
                  {skipPaths.map((p, i) => <path key={'k'+i} className={p.type === 'unet' ? 'skip-path-unet' : 'skip-path'} d={p.d} markerEnd={`url(#arrowhead-${p.type})`} />)}
                </svg>
                
                <div className={`pipeline ${mode === 'unet' ? 'unet-layout' : ''}`}>
                  {!graph.length && <div className="empty">Drag & drop architecture blocks here</div>}
                  
                  {graph.map((block, idx) => {
                    const gridStyle = {};
                    if (mode === "unet") {
                      if (block.kind.startsWith("u_enc") || block.kind === "u_input") {
                        gridStyle.gridColumn = 1; gridStyle.gridRow = currentEnc++;
                      } else if (block.kind.includes("bottleneck") || block.kind.includes("bridge")) {
                        gridStyle.gridColumn = 2; gridStyle.gridRow = Math.max(currentEnc, 2);
                      } else {
                        gridStyle.gridColumn = 3; gridStyle.gridRow = Math.max(1, currentEnc - currentDec);
                        currentDec++;
                      }
                    }

                    return (
                      <div key={block.id} className="node-wrapper" style={gridStyle}>
                        {dropTargetIdx === idx && <div className="drop-indicator"/>}
                        <article 
                          ref={el => el ? nodeRefs.current.set(block.id, el) : nodeRefs.current.delete(block.id)}
                          className={`node ${mode} ${selectedId === block.id ? "selected" : ""} ${issues.get(block.id)==="error"?"v-error":issues.get(block.id)==="warning"?"v-warning":""} ${draggedId===block.id?"dragging":""}`}
                          onClick={() => setSelectedId(block.id)}
                          draggable
                          onDragStart={e => { e.dataTransfer.setData("app/id", block.id); setDraggedId(block.id); }}
                          onDragEnd={() => { setDraggedId(null); setDropTargetIdx(null); }}
                          onDragOver={e => handleDropTarget(e, idx)}
                          onDrop={e => handleDrop(e, idx)}
                        >
                          <div className="node-top">
                            <div className="drag-handle"><span/><span/><span/></div>
                            <span className="glyph">{block.glyph}</span>
                            <div><strong>{block.label}</strong><small>{block.kind}</small></div>
                            <button className="icon-btn" onClick={e=>{e.stopPropagation(); updateGraph(graph.filter(b=>b.id!==block.id))}}><Trash2 size={16}/></button>
                          </div>
                          <div className="node-body">
                            {mode !== "ml" && shapes[idx] && (
                              <span className="chip dim-flow">
                                {shapes[idx].h_in}×{shapes[idx].w_in}×{shapes[idx].c_in} → {shapes[idx].h}×{shapes[idx].w}×{shapes[idx].c}
                              </span>
                            )}
                            {block.attention?.map(a => <span key={a.id} className="chip attn">{a.label}</span>)}
                          </div>
                        </article>
                      </div>
                    );
                  })}
                  {dropTargetIdx === graph.length && <div className="drop-indicator" style={{bottom: '-20px', top: 'auto'}}/>}
                </div>
              </div>
            </div>
          </section>
        )}

        {view !== "builder" && (
          <section className="panel" style={{gridColumn: "2 / span 2"}}>
            {view === "data" && (
              <div className="scroll">
                <h2>Compatible Datasets</h2>
                <div className="dataset-grid">
                  {(mode === "ml" ? csvDatasets : imageDatasets)
                    .filter(d => d.compatible.includes(mode))
                    .map(d => (
                    <div key={d.name} className={`dataset-card ${dataset.name === d.name ? "active" : ""}`} onClick={() => setDataset(d)}>
                      <div><strong>{d.name}</strong> <a href={d.url} target="_blank" rel="noreferrer">View Source</a></div>
                      <small>{d.category} • {d.classes||d.target} • {d.size||d.samples}</small>
                    </div>
                  ))}
                </div>
                {mode !== "ml" && (
                  <div style={{marginTop: 24}}>
                    <h2>Image Augmentations</h2>
                    <div className="check-grid">
                      {augmentations.map(a => (
                        <label key={a} className="check">
                          <input type="checkbox" checked={selectedAugs.includes(a)} onChange={() => setSelectedAugs(p => p.includes(a)?p.filter(x=>x!==a):[...p,a])}/> {a}
                        </label>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
            
            {view === "loss" && (
              <div className="scroll">
                <h2>Loss Functions</h2>
                <div className="loss-grid">
                  {lossCatalog[mode].map(l => (
                    <div key={l.kind} className={`loss-card ${losses[mode].kind === l.kind ? "active" : ""}`} onClick={() => setLosses(p => ({...p, [mode]: l}))}>
                      <div><strong>{l.label}</strong><small>{l.meta}</small></div>
                    </div>
                  ))}
                </div>
                
                {losses[mode] && Object.keys(losses[mode].params).length > 0 && (
                  <div style={{marginTop: 24}}>
                    <h2>Hyperparameters</h2>
                    <div className="form-grid">
                      {Object.entries(losses[mode].params).map(([k,v]) => (
                        <label key={k}>{k} <input type="number" step="0.1" value={v} onChange={e => setLosses(p => ({...p, [mode]: {...p[mode], params:{...p[mode].params, [k]: Number(e.target.value)}}}) )}/></label>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
            
            {view === "code" && <textarea className="code" value={generatedCode} readOnly spellCheck="false" />}
          </section>
        )}

        {view === "builder" && (
          <aside className="panel inspector-panel">
            <div className="panel-header"><div><span>Inspector</span><strong>Block Settings</strong></div></div>
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
                              <select value={selected.params[k]} onChange={e => updateGraph(graph.map(b=>b.id===selected.id?{...b,params:{...b.params,[k]:e.target.value}}:b))}>
                                {(k==="activation"?activations:["nearest","bilinear"]).map(o=><option key={o} value={o}>{o}</option>)}
                              </select>
                            ) : (
                              <input type={typeof selected.params[k]==="number"?"number":"text"} value={selected.params[k]} onChange={e => updateGraph(graph.map(b=>b.id===selected.id?{...b,params:{...b.params,[k]:typeof selected.params[k]==="number"?Number(e.target.value):e.target.value}}:b))}/>
                            )}
                          </label>
                        )
                      ))}
                      {(mode === "cnn" && residualKinds.has(selected.kind)) && (
                         <label className="wide">Skip Connection Target
                           <select value={selected.params.skipTo||""} onChange={e => updateGraph(graph.map(b=>b.id===selected.id?{...b,params:{...b.params,skipTo:e.target.value}}:b))}>
                             <option value="">No Skip (Next Block)</option>
                             {graph.slice(graph.findIndex(b=>b.id===selected.id)+1).map(b => <option key={b.id} value={b.id}>{b.label}</option>)}
                           </select>
                         </label>
                      )}
                    </div>
                  </div>
                </>
              )}
            </div>
          </aside>
        )}
      </main>
      <div className={`toast ${toast?"show":""}`}>{toast}</div>
    </div>
  );
}
