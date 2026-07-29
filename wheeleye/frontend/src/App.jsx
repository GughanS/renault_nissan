import { useState, useEffect, useRef } from 'react'
import { Play, Square, Upload } from 'lucide-react'
import mockData from './mocks/inspect.json'
import './index.css'

// Tier → expected fastener count mapping
const TIER_FASTENER_MAP = {
  Standard: 4,
  Premium: 5,
  Luxury: 6
};

function App() {
  const { demo_frames } = mockData;
  
  const [isDemoRunning, setIsDemoRunning] = useState(false);
  const [taktTime, setTaktTime] = useState(1500);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  
  // Dynamic SKU selector state
  const [skuMaterial, setSkuMaterial] = useState('Alloy');
  const [skuTier, setSkuTier] = useState('Premium');
  const [skuSize, setSkuSize] = useState('18_inch');
  
  // Real backend state
  const [realReport, setRealReport] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("https://placehold.co/512x512/1C1C1E/3A3A3C/png?text=Live+Camera+Feed");
  
  // Use a fallback stats object when starting up
  const initialStats = {
    station_id: "ST-42B",
    uptime: "000:00:00",
    units_inspected: "0",
    pass_rate: "100.0%",
    avg_latency: "0.0ms"
  };
  
  const [stats, setStats] = useState(initialStats);
  
  const [logs, setLogs] = useState([
    {
      timestamp: "08:42:10.650",
      status: "PASS",
      details: "Assembly verified successfully.",
      thumbnail: "https://placehold.co/100x100/242424/34C759/png?text=OK"
    }
  ]);
  
  const logTableRef = useRef(null);
  const fileInputRef = useRef(null);

  // Compute expected fasteners from tier selection
  const expectedFasteners = TIER_FASTENER_MAP[skuTier] || 5;

  // Auto-scroll logs
  useEffect(() => {
    if (logTableRef.current) {
      logTableRef.current.scrollTop = 0;
    }
  }, [logs]);

  // Demo Cycle Logic
  useEffect(() => {
    let intervalId;
    if (isDemoRunning) {
      // Clear real report when entering demo mode
      setRealReport(null);
      setPreviewUrl("https://placehold.co/512x512/1C1C1E/3A3A3C/png?text=Live+Camera+Feed");
      
      intervalId = setInterval(() => {
        setCurrentFrameIndex((prevIndex) => {
          const nextIndex = (prevIndex + 1) % demo_frames.length;
          const nextFrame = demo_frames[nextIndex];
          
          const now = new Date();
          const timestamp = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}.${now.getMilliseconds().toString().padStart(3, '0')}`;
          
          const newLog = {
            timestamp,
            status: nextFrame.status,
            details: nextFrame.status === 'PASS' ? 'Assembly verified successfully.' : nextFrame.messages[0],
            thumbnail: nextFrame.thumbnail
          };
          
          setLogs(prevLogs => [newLog, ...prevLogs].slice(0, 50));
          
          // Fake stats update in demo mode
          setStats(prev => ({
            ...prev,
            units_inspected: (parseInt(prev.units_inspected.replace(/,/g, '')) + 1).toLocaleString(),
            avg_latency: "14.2ms"
          }));
          
          return nextIndex;
        });
      }, taktTime);
    }
    return () => clearInterval(intervalId);
  }, [isDemoRunning, taktTime, demo_frames]);

  // Handle Real File Upload
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    // Stop demo if running
    setIsDemoRunning(false);
    setIsUploading(true);
    setUploadError(null);
    
    // Generate local preview URL
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
    
    const formData = new FormData();
    formData.append('file', file);
    
    // Build manifest dynamically from the SKU selector dropdowns
    const manifest = {
      material: skuMaterial,
      tier: skuTier,
      size: skuSize,
      expected_fasteners: expectedFasteners
    };
    formData.append('manifest', JSON.stringify(manifest));

    try {
      const response = await fetch('http://localhost:8000/inspect', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }
      
      const data = await response.json();
      setRealReport(data);
      
      if (data.stats) {
        setStats(data.stats);
      }
      
      // Add to logs
      const now = new Date();
      const timestamp = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}.${now.getMilliseconds().toString().padStart(3, '0')}`;
      
      const newLog = {
        timestamp,
        status: data.status,
        details: data.status === 'PASS' ? 'Assembly verified successfully.' : (data.messages[0] || 'Unknown error'),
        thumbnail: data.thumbnail || "https://placehold.co/100x100/242424/34C759/png?text=OK"
      };
      
      setLogs(prevLogs => [newLog, ...prevLogs].slice(0, 50));
      
    } catch (err) {
      console.error(err);
      setUploadError("Failed to reach inference server. Is the FastAPI backend running?");
    } finally {
      setIsUploading(false);
      // Reset input so same file can be uploaded again if needed
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  // Determine which report to show
  const currentReport = isDemoRunning ? demo_frames[currentFrameIndex] : (realReport || {
    status: "STANDBY",
    detections: [],
    classification: { material: "---", tier: "---", size: "---" },
    messages: []
  });
  
  const getClassColor = (className) => {
    switch(className) {
      case 'wheel': return 'var(--color-utility-blue)';
      case 'fastener': return 'var(--color-utility-blue)';
      case 'scratch': return 'var(--color-fail-red)';
      case 'dent': return 'var(--color-fail-red)';
      default: return 'var(--color-utility-blue)';
    }
  }

  // BBox logic. If we have a real image, we use its intrinsic dimensions.
  // We'll calculate percentages using the box coords assuming the model uses 512x512 inputs and outputs boxes in that scale.
  // Wait, real YOLOv8 outputs boxes in the original image coordinate space usually, unless it was resized.
  // Let's assume the backend provides coordinates in the original image scale.
  // But wait, our verifier runs YOLO on the original image, so the boxes are in the original image's pixels.
  // We need to know the original image size to convert to percentages.
  // As a hack for the UI, let's just assume the image gets rendered via object-fit contain, and CSS percentages based on the container won't exactly match if aspect ratio differs.
  // For a robust system, the backend should return the original image dimensions. We will hardcode 512 for the demo frames and 640 for real frames for now, or just use 640.
  const baseDim = isDemoRunning ? 512 : 640; 
  
  const renderBBox = (det, index) => {
    const [x1, y1, x2, y2] = det.box;
    const width = x2 - x1;
    const height = y2 - y1;
    
    // We assume 640x640 is the standard camera feed size
    const left = `${(x1 / baseDim) * 100}%`;
    const top = `${(y1 / baseDim) * 100}%`;
    const wPct = `${(width / baseDim) * 100}%`;
    const hPct = `${(height / baseDim) * 100}%`;
    
    const color = getClassColor(det.class_name);
    const isDefect = det.class_name === 'scratch' || det.class_name === 'dent';
    
    return (
      <div 
        key={index}
        className={`bbox ${isDefect ? 'bbox-defect' : ''}`}
        style={{
          left, top, width: wPct, height: hPct,
          borderColor: color
        }}
      >
        <div className="bbox-label" style={{ backgroundColor: color }}>
          {det.class_name.toUpperCase()} {det.score.toFixed(2)}
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      {/* Top Summary Strip */}
      <div className="status-strip">
        <div className="group">
          <div><span className="label">STATION</span> <span className="value mono">{stats.station_id}</span></div>
          <div><span className="label">UPTIME</span> <span className="value mono">{stats.uptime}</span></div>
        </div>
        <div className="group">
          <div><span className="label">TOTAL UNITS</span> <span className="value mono">{stats.units_inspected}</span></div>
          <div><span className="label">PASS RATE</span> <span className="value mono" style={{color: 'var(--color-pass-green)'}}>{stats.pass_rate}</span></div>
          <div><span className="label">AVG LATENCY</span> <span className="value mono">{stats.avg_latency}</span></div>
          
          {/* Demo & Upload Controls */}
          <div className="demo-controls">
            <input 
              type="file" 
              accept="image/*" 
              style={{display: 'none'}} 
              ref={fileInputRef}
              onChange={handleFileUpload}
            />
            <button 
              className="demo-toggle"
              style={{ borderColor: 'var(--color-border)', color: 'var(--color-text-main)' }}
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading}
            >
              <Upload size={14} />
              {isUploading ? 'Inspecting...' : 'Manual Upload'}
            </button>
            <select 
              className="takt-time-select"
              value={taktTime}
              onChange={(e) => setTaktTime(Number(e.target.value))}
            >
              <option value={3000}>Takt: 3.0s</option>
              <option value={1500}>Takt: 1.5s</option>
              <option value={500}>Takt: 0.5s</option>
            </select>
            <button 
              className={`demo-toggle ${isDemoRunning ? 'active' : ''}`}
              onClick={() => setIsDemoRunning(!isDemoRunning)}
            >
              {isDemoRunning ? <Square size={14} /> : <Play size={14} fill="currentColor" />}
              {isDemoRunning ? 'Stop Sim' : 'Simulate Line'}
            </button>
          </div>
        </div>
      </div>

      {/* Main Layout */}
      <div className="main-layout">
        <div className="left-panel">
          <div className="inspection-view">
            <div className="crosshair-h"></div>
            <div className="crosshair-v"></div>
            
            {uploadError ? (
              <div style={{ color: 'var(--color-fail-red)', padding: '20px', textAlign: 'center' }}>
                <h3>ERROR</h3>
                <p>{uploadError}</p>
              </div>
            ) : (
              <div style={{ position: 'relative', display: 'block', aspectRatio: '1 / 1', maxWidth: '100%', maxHeight: '100%', margin: '0 auto' }}>
                <img src={previewUrl} alt="Live Inspection Frame" style={{ display: 'block', width: '100%', height: '100%', objectFit: 'cover', opacity: isUploading ? 0.3 : 1 }} />
                {(() => {
                  if (currentReport.status === "STANDBY" || isUploading || currentReport.status === "PASS") return null;
                  
                  const msgStr = (currentReport.messages || []).join(" ").toLowerCase();
                  const showDefects = msgStr.includes("defect");
                  const showFasteners = msgStr.includes("fastener");
                  const showWheel = msgStr.includes("wrong");
                  
                  return currentReport.detections.filter(det => {
                    if ((det.class_name === 'scratch' || det.class_name === 'dent') && showDefects) return true;
                    if (det.class_name === 'fastener' && showFasteners) return true;
                    if (det.class_name === 'wheel' && showWheel) return true;
                    return false;
                  }).map(renderBBox);
                })()}
                
                {isUploading && (
                  <div className="loading-overlay">
                    <div className="spinner"></div>
                    <div className="loading-text mono">INSPECTING...</div>
                  </div>
                )}
              </div>
            )}
          </div>
          
          <div className="inspection-log">
            <div className="log-header">
              Recent Inspections
            </div>
            <div className="log-table" ref={logTableRef}>
              {logs.map((entry, idx) => (
                <div key={idx} className="log-row">
                  <div className="time mono">{entry.timestamp}</div>
                  <div className="thumb"><img src={entry.thumbnail} alt="thumb" /></div>
                  <div className={`status ${entry.status.toLowerCase()}`}>{entry.status}</div>
                  <div className="details">{entry.details}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="right-panel">
          {/* SKU Selector Panel */}
          <div className="sku-selector">
            <div className="sku-title">Incoming Wheel SKU</div>
            <div className="sku-row">
              <label className="sku-label">Material</label>
              <select
                id="sku-material"
                className="sku-select"
                value={skuMaterial}
                onChange={(e) => setSkuMaterial(e.target.value)}
              >
                <option value="Steel">Steel</option>
                <option value="Alloy">Alloy</option>
              </select>
            </div>
            <div className="sku-row">
              <label className="sku-label">Tier</label>
              <select
                id="sku-tier"
                className="sku-select"
                value={skuTier}
                onChange={(e) => setSkuTier(e.target.value)}
              >
                <option value="Standard">Standard</option>
                <option value="Premium">Premium</option>
                <option value="Luxury">Luxury</option>
              </select>
            </div>
            <div className="sku-row">
              <label className="sku-label">Size</label>
              <select
                id="sku-size"
                className="sku-select"
                value={skuSize}
                onChange={(e) => setSkuSize(e.target.value)}
              >
                <option value="17_inch">17"</option>
                <option value="18_inch">18"</option>
                <option value="19_inch">19"</option>
              </select>
            </div>
            <div className="sku-fastener-info mono">
              Expected fasteners: {expectedFasteners}
            </div>
          </div>

          <div className="verdict-panel">
            <div className="verdict-title">Current Assembly Verdict</div>
            <div className={`verdict-block ${currentReport.status.toLowerCase()}`}>
              {currentReport.status}
            </div>
          </div>
          
          <div className="analysis-details">
            <div className="detail-group">
              <div className="detail-label">Classification</div>
              <div className="chips">
                <div className="chip mono">{currentReport.classification.material.toUpperCase()}</div>
                <div className="chip mono">{currentReport.classification.tier.toUpperCase()}</div>
                <div className="chip mono">{currentReport.classification.size.toUpperCase()}</div>
              </div>
            </div>
            
            <div className="detail-group">
              <div className="detail-label">Fastener Count</div>
              <div className="fastener-count mono">
                {currentReport.status === "STANDBY" ? `0 / ${expectedFasteners}` : `${currentReport.detections.filter(d => d.class_name === 'fastener').length} / ${expectedFasteners}`}
              </div>
            </div>
            
            {currentReport.messages && currentReport.messages.length > 0 && currentReport.status === 'FAIL' && (
              <div className="detail-group">
                <div className="detail-label" style={{color: 'var(--color-fail-red)'}}>Errors</div>
                <ul className="messages-list">
                  {currentReport.messages.map((msg, idx) => (
                    <li key={idx}>{msg}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
