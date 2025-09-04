import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

function getRadiusByScore(score, maxScore) {
  // Scale the radius based on the score relative to the maximum score
  // Adjusted range for better visual balance
  const minRadius = 30;
  const maxRadius = 65;
  const normalizedScore = score / maxScore;
  return minRadius + (normalizedScore * (maxRadius - minRadius));
}

function getOptimalFontSize(text, radius) {
  // Calculate optimal font size based on text length and bubble radius
  const textLength = text.length;
  const maxWidth = radius * 1.8; // Increased padding (90% of diameter)
  
  // Base font size calculation - increased base size
  let fontSize = Math.min(radius * 0.6, maxWidth / (textLength * 0.5));
  
  // Ensure minimum and maximum bounds - increased minimum
  fontSize = Math.max(10, Math.min(fontSize, 20));
  
  // For very long text, reduce font size less aggressively
  if (textLength > 15) {
    fontSize = Math.max(8, fontSize * 0.85);
  }
  
  // For very short text, allow larger font
  if (textLength <= 5) {
    fontSize = Math.min(fontSize * 1.3, 22);
  }
  
  // For medium length text, boost the size
  if (textLength > 5 && textLength <= 10) {
    fontSize = Math.min(fontSize * 1.1, 18);
  }
  
  return Math.round(fontSize);
}

function PackedBubbleChart({ data = [] }) {
  const svgRef = useRef();
  const [hoveredIdx, setHoveredIdx] = useState(null);
  const [nodes, setNodes] = useState([]);

  useEffect(() => {
    if (!data || data.length === 0) {
      setNodes([]);
      return;
    }
    const width = 800, height = 800;
    const maxScore = Math.max(...data.map(d => d.score));
    const root = d3.hierarchy({ children: data }).sum(d => getRadiusByScore(d.score, maxScore) ** 2);
    d3.pack().size([width, height]).padding(12)(root);
    setNodes(root.descendants());
  }, [data]);

  // hover된 노드를 맨 뒤로 빼서 맨 위에 렌더링
  const orderedNodes = hoveredIdx !== null && nodes.length > 0
    ? [
        ...nodes.filter((_, i) => i !== hoveredIdx),
        nodes[hoveredIdx]
      ]
    : nodes;

  return (
    <svg ref={svgRef} width={800} height={800} style={{ display: 'block', margin: '0 auto' }}>
      <defs>
        <linearGradient id="bubbleLinear" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#ff6ac1" stopOpacity="1" />
          <stop offset="100%" stopColor="#38bdf8" stopOpacity="1" />
        </linearGradient>
        <filter id="textShadow" x="-50%" y="-50%" width="200%" height="200%">
          <feDropShadow dx="2" dy="2" stdDeviation="2" floodColor="#000" floodOpacity="0.4" />
        </filter>
        <filter id="bubbleNeon" x="-60%" y="-60%" width="220%" height="220%">
          <feDropShadow dx="0" dy="0" stdDeviation="8" floodColor="#c084fc" floodOpacity="0.85" />
          <feDropShadow dx="0" dy="0" stdDeviation="16" floodColor="#a259f7" floodOpacity="0.5" />
          <feDropShadow dx="0" dy="0" stdDeviation="32" floodColor="#fff" floodOpacity="0.18" />
        </filter>
      </defs>
      

      {orderedNodes.map((d) => (
        <g
          key={d.data.keyword || d.data.name}
          transform={`translate(${d.x},${d.y}) scale(${hoveredIdx !== null && nodes[hoveredIdx] === d ? 1.12 : (hoveredIdx !== null && nodes.indexOf(d) === hoveredIdx && d.depth !== 0 ? 1.12 : 1)})`}
          style={{ transition: 'transform 0.2s', cursor: d.depth !== 0 ? 'pointer' : 'default' }}
          onMouseOver={() => d.depth !== 0 && setHoveredIdx(nodes.indexOf(d))}
          onMouseOut={() => setHoveredIdx(null)}
        >
          <circle
            r={d.depth !== 0 ? getRadiusByScore(d.data.score, Math.max(...data.map(d => d.score))) : 0}
            fill={d.depth === 0 ? "none" : "url(#bubbleLinear)"}
            opacity={d.depth === 0 ? 1 : 1}
            stroke={d.depth !== 0 ? "rgba(255, 255, 255, 0.3)" : "none"}
            strokeWidth={d.depth !== 0 ? "2" : "0"}
            filter={d.depth !== 0 ? "url(#bubbleNeon)" : undefined}
          />
          {d.depth !== 0 && (
            <text
              textAnchor="middle"
              dy=".35em"
              filter="url(#textShadow)"
              style={{
                fontSize: getOptimalFontSize(d.data.keyword || d.data.name, getRadiusByScore(d.data.score, Math.max(...data.map(d => d.score)))),
                fill: "#fff",
                pointerEvents: "none",
                fontFamily: 'Inter, sans-serif',
                fontWeight: '600'
              }}
            >
              {d.data.keyword || d.data.name}
            </text>
          )}
        </g>
      ))}
    </svg>
  );
}

export default PackedBubbleChart; 