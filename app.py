# ============================================================
# app.py — MESAD Surgical Action Detection (Enhanced)
# Save this file and run: streamlit run app.py
# ============================================================
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
from pathlib import Path
import gdown
from ultralytics import YOLO

#https://drive.google.com/file/d/1XW9rN5IsoUrDXeJsZ8ORTB8BcLOITLdw/view?usp=sharing
# ── Page config ──
st.set_page_config(
    page_title="MESAD — Surgical Action Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme toggle state ──
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# ── Custom CSS ──
def get_css(dark):
    if dark:
        return """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Sora:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

.main, [data-testid="stAppViewContainer"] {
    background: #080d1a;
    color: #dde2f0;
}
[data-testid="stSidebar"] {
    background: #0c1222;
    border-right: 1px solid #1a2540;
}
[data-testid="stSidebar"] * { color: #dde2f0 !important; }

.stButton > button {
    background: #1a56db;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1.4rem;
    font-family: 'Sora', sans-serif;
    font-weight: 500;
    font-size: 14px;
    width: 100%;
    transition: background 0.2s, transform 0.1s;
}
.stButton > button:hover { background: #1e64f0; transform: translateY(-1px); }

.metric-card {
    background: #0c1222;
    border: 1px solid #1a2540;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.65rem;
}
.metric-card .label {
    font-size: 10px; color: #4a5a7a;
    text-transform: uppercase; letter-spacing: 0.1em;
    font-family: 'JetBrains Mono', monospace; margin-bottom: 4px;
}
.metric-card .value {
    font-size: 26px; font-weight: 600; color: #dde2f0;
    font-family: 'JetBrains Mono', monospace;
}
.metric-card .sub { font-size: 11px; color: #2a6bd5; margin-top: 2px; }

.header-title {
    font-size: 30px; font-weight: 600; color: #dde2f0;
    letter-spacing: -0.02em; margin-bottom: 3px;
}
.header-sub {
    font-size: 13px; color: #4a5a7a;
    font-family: 'JetBrains Mono', monospace; margin-bottom: 1.5rem;
}
.section-label {
    font-size: 10px; color: #4a5a7a;
    text-transform: uppercase; letter-spacing: 0.1em;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.65rem; padding-bottom: 0.4rem;
    border-bottom: 1px solid #1a2540;
}
.class-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 0; border-bottom: 1px solid #0c1222; font-size: 13px;
}
.class-row .cls-name { color: #9aacc8; font-family: 'JetBrains Mono', monospace; font-size: 12px; }
.conf-bar-wrap { flex: 1; margin: 0 10px; height: 4px; background: #1a2540; border-radius: 2px; }
.conf-bar { height: 4px; border-radius: 2px; background: #1a56db; }
.conf-val { color: #2a6bd5; font-family: 'JetBrains Mono', monospace; font-size: 12px; min-width: 38px; text-align: right; }

div[data-testid="stImage"] img { border-radius: 10px; }

/* Streamlit tab styling */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent; border-bottom: 1px solid #1a2540; gap: 0;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    color: #4a5a7a; font-family: 'Sora', sans-serif; font-size: 13px;
    padding: 0.5rem 1.2rem; border-radius: 0;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #1a56db !important; border-bottom: 2px solid #1a56db !important;
    background: transparent !important;
}

/* Table override */
[data-testid="stDataFrame"] table {
    background: #0c1222 !important; color: #dde2f0 !important;
}
[data-testid="stDataFrame"] th {
    background: #1a2540 !important; color: #9aacc8 !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important;
}
[data-testid="stDataFrame"] td {
    color: #dde2f0 !important; font-size: 13px !important;
    background: #0c1222 !important;
}
[data-testid="stDataFrame"] tr:hover td { background: #1a2540 !important; }

/* Info/success/error boxes */
.stAlert { border-radius: 8px !important; }

/* Slider labels */
[data-testid="stSlider"] label { color: #9aacc8 !important; }

/* Plotly chart borders */
.js-plotly-plot { border-radius: 10px; overflow: hidden; }
</style>"""
    else:
        return """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Sora:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

.main, [data-testid="stAppViewContainer"] {
    background: #f4f6fb;
    color: #1a2540;
}
[data-testid="stSidebar"] {
    background: #ebeef7;
    border-right: 1px solid #cdd5e8;
}
[data-testid="stSidebar"] * { color: #1a2540 !important; }

.stButton > button {
    background: #1a56db;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1.4rem;
    font-family: 'Sora', sans-serif;
    font-weight: 500;
    font-size: 14px;
    width: 100%;
    transition: background 0.2s, transform 0.1s;
}
.stButton > button:hover { background: #1e64f0; transform: translateY(-1px); }

.metric-card {
    background: #ffffff;
    border: 1px solid #cdd5e8;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.65rem;
    box-shadow: 0 1px 4px rgba(26,37,64,0.07);
}
.metric-card .label {
    font-size: 10px; color: #7a8aaa;
    text-transform: uppercase; letter-spacing: 0.1em;
    font-family: 'JetBrains Mono', monospace; margin-bottom: 4px;
}
.metric-card .value {
    font-size: 26px; font-weight: 600; color: #1a2540;
    font-family: 'JetBrains Mono', monospace;
}
.metric-card .sub { font-size: 11px; color: #1a56db; margin-top: 2px; }

.header-title {
    font-size: 30px; font-weight: 600; color: #1a2540;
    letter-spacing: -0.02em; margin-bottom: 3px;
}
.header-sub {
    font-size: 13px; color: #7a8aaa;
    font-family: 'JetBrains Mono', monospace; margin-bottom: 1.5rem;
}
.section-label {
    font-size: 10px; color: #7a8aaa;
    text-transform: uppercase; letter-spacing: 0.1em;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.65rem; padding-bottom: 0.4rem;
    border-bottom: 1px solid #cdd5e8;
}
.class-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 0; border-bottom: 1px solid #ebeef7; font-size: 13px;
}
.class-row .cls-name { color: #3a5080; font-family: 'JetBrains Mono', monospace; font-size: 12px; }
.conf-bar-wrap { flex: 1; margin: 0 10px; height: 4px; background: #cdd5e8; border-radius: 2px; }
.conf-bar { height: 4px; border-radius: 2px; background: #1a56db; }
.conf-val { color: #1a56db; font-family: 'JetBrains Mono', monospace; font-size: 12px; min-width: 38px; text-align: right; }

div[data-testid="stImage"] img { border-radius: 10px; box-shadow: 0 2px 12px rgba(26,37,64,0.10); }

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent; border-bottom: 1px solid #cdd5e8; gap: 0;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    color: #7a8aaa; font-family: 'Sora', sans-serif; font-size: 13px;
    padding: 0.5rem 1.2rem; border-radius: 0;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #1a56db !important; border-bottom: 2px solid #1a56db !important;
    background: transparent !important;
}

[data-testid="stDataFrame"] table { background: #fff !important; color: #1a2540 !important; }
[data-testid="stDataFrame"] th {
    background: #ebeef7 !important; color: #3a5080 !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important;
}
[data-testid="stDataFrame"] td {
    color: #1a2540 !important; font-size: 13px !important; background: #fff !important;
}
[data-testid="stDataFrame"] tr:hover td { background: #f4f6fb !important; }

.stAlert { border-radius: 8px !important; }
[data-testid="stSlider"] label { color: #3a5080 !important; }
.js-plotly-plot { border-radius: 10px; overflow: hidden; }
</style>"""

st.markdown(get_css(st.session_state.dark_mode), unsafe_allow_html=True)


MODEL_PATH = "best_combined_s.pt"
FILE_ID = "1XW9rN5IsoUrDXeJsZ8ORTB8BcLOITLdw"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

@st.cache_resource
def load_model(model_path):
    # Download model if not present
    if not os.path.exists(model_path):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(URL, model_path, quiet=False)
    return YOLO(model_path)

# ── Load model ──
model = None
with st.spinner("Loading model..."):
    model = load_model(MODEL_PATH)
st.success(f"Model loaded: `{MODEL_PATH}`")


# ── Class names ──
CLASS_NAMES = [
    'BaggingProstate', 'BladderAnastomosis', 'BladderNeckDissection',
    'ClippingBladderNeck', 'ClippingSeminalVesicle', 'ClippingTissue',
    'ClippingVasDeferens', 'CuttingMesocolon', 'CuttingProstate',
    'CuttingSeminalVesicle', 'CuttingThread', 'CuttingTissue',
    'CuttingVasDeferens', 'GraspingCatheter', 'MovingDownBladder',
    'PassingNeedle', 'PullingBladderNeck', 'PullingProstate',
    'PullingSeminalVesicle', 'PullingTissue', 'PullingVasDeferens',
    'SuckingBlood', 'SuckingSmoke', 'UrethraDissection'
]

# ── Color palette for charts ──
CHART_COLORS = [
    '#1a56db', '#0ea5e9', '#06b6d4', '#14b8a6', '#10b981',
    '#84cc16', '#f59e0b', '#f97316', '#ef4444', '#ec4899',
    '#8b5cf6', '#6366f1', '#3b82f6', '#0284c7', '#0369a1',
    '#065f46', '#92400e', '#7c3aed', '#be185d', '#0891b2',
    '#059669', '#d97706', '#dc2626', '#9333ea'
]

# ── Plotly chart helper ──
def make_bar_chart(names, values, title, xlabel, ylabel, dark):
    import plotly.graph_objects as go
    bg     = '#0c1222' if dark else '#ffffff'
    paper  = '#080d1a' if dark else '#f4f6fb'
    txt    = '#dde2f0' if dark else '#1a2540'
    grid   = '#1a2540' if dark else '#e2e8f0'
    colors = [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(names))]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=[f'{v:.3f}' if isinstance(v, float) else str(v) for v in values],
        textposition='outside',
        textfont=dict(color=txt, size=11, family='JetBrains Mono'),
        hovertemplate='<b>%{y}</b><br>%{x}<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=txt, family='Sora'), x=0),
        plot_bgcolor=bg, paper_bgcolor=paper,
        font=dict(color=txt, family='Sora'),
        xaxis=dict(title=xlabel, gridcolor=grid, zerolinecolor=grid,
                   tickfont=dict(size=10, family='JetBrains Mono')),
        yaxis=dict(title=ylabel, gridcolor=grid, tickfont=dict(size=10, family='JetBrains Mono'),
                   autorange='reversed'),
        margin=dict(l=10, r=60, t=40, b=40),
        height=max(300, len(names) * 32 + 80),
        bargap=0.35,
    )
    return fig

def make_pie_chart(names, values, title, dark):
    import plotly.graph_objects as go
    bg    = '#0c1222' if dark else '#ffffff'
    paper = '#080d1a' if dark else '#f4f6fb'
    txt   = '#dde2f0' if dark else '#1a2540'

    fig = go.Figure(go.Pie(
        labels=names, values=values,
        marker=dict(colors=CHART_COLORS[:len(names)], line=dict(color=bg, width=2)),
        textfont=dict(size=11, family='JetBrains Mono', color='white'),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>',
        hole=0.4,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=txt, family='Sora'), x=0),
        plot_bgcolor=bg, paper_bgcolor=paper,
        font=dict(color=txt, family='Sora'),
        legend=dict(font=dict(size=10, family='JetBrains Mono', color=txt),
                    bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=10, r=10, t=40, b=10),
        height=400,
    )
    return fig

def make_timeline_chart(frame_nums, conf_per_frame, title, dark):
    """Line chart of max confidence across frames (video)."""
    import plotly.graph_objects as go
    bg    = '#0c1222' if dark else '#ffffff'
    paper = '#080d1a' if dark else '#f4f6fb'
    txt   = '#dde2f0' if dark else '#1a2540'
    grid  = '#1a2540' if dark else '#e2e8f0'

    fig = go.Figure(go.Scatter(
        x=frame_nums, y=conf_per_frame,
        mode='lines',
        line=dict(color='#1a56db', width=2),
        fill='tozeroy',
        fillcolor='rgba(26,86,219,0.12)',
        hovertemplate='Frame %{x}<br>Max conf: %{y:.3f}<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=txt, family='Sora'), x=0),
        plot_bgcolor=bg, paper_bgcolor=paper,
        font=dict(color=txt, family='Sora'),
        xaxis=dict(title='Frame', gridcolor=grid, tickfont=dict(size=10, family='JetBrains Mono')),
        yaxis=dict(title='Max Confidence', gridcolor=grid, range=[0, 1],
                   tickfont=dict(size=10, family='JetBrains Mono')),
        margin=dict(l=10, r=10, t=40, b=40),
        height=280,
    )
    return fig

# ── Sidebar ──
with st.sidebar:
    # Theme toggle
    dm_label = "☀️ Light mode" if st.session_state.dark_mode else "🌙 Dark mode"
    if st.button(dm_label):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown('<div class="section-label" style="margin-top:1rem">Model</div>',
                unsafe_allow_html=True)
    model_path = st.text_input("Weights path", value="best_combined_s.pt",
                               label_visibility="collapsed")

    st.markdown('<div class="section-label" style="margin-top:1rem">Settings</div>',
                unsafe_allow_html=True)
    conf_thresh = st.slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05)
    iou_thresh  = st.slider("IoU threshold",         0.10, 0.90, 0.45, 0.05)
    show_labels = st.checkbox("Show labels",      value=True)
    show_conf   = st.checkbox("Show confidence",  value=True)

    st.markdown('<div class="section-label" style="margin-top:1.5rem">Model info</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Architecture</div>
        <div class="value" style="font-size:16px">YOLOv8s</div>
        <div class="sub">Combined dataset</div>
    </div>
    <div class="metric-card">
        <div class="label">Classes</div>
        <div class="value">{len(CLASS_NAMES)}</div>
        <div class="sub">Surgical actions</div>
    </div>
    <div class="metric-card">
        <div class="label">mAP@0.5</div>
        <div class="value">0.146</div>
        <div class="sub">Step 4 baseline</div>
    </div>
    """, unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div class="header-title">🔬 MESAD — Surgical Action Detection</div>
<div class="header-sub">Multi-domain endoscopic surgical action recognition · YOLOv8s · 24 classes</div>
""", unsafe_allow_html=True)

# ── Load model ──
model = None
if os.path.exists(model_path):
    with st.spinner("Loading model..."):
        model = load_model(model_path)
    st.success(f"Model loaded: `{model_path}`")
else:
    st.error(f"Model not found at `{model_path}` — update the path in the sidebar")
    st.stop()

dark = st.session_state.dark_mode

# ── Tabs ──
tab1, tab2, tab3 = st.tabs(["🖼️  Image", "🎬  Video", "📂  Batch"])

# ============================================================
# TAB 1 — Single image
# ============================================================
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload surgical frame", type=["jpg","jpeg","png"],
            label_visibility="collapsed"
        )
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, use_column_width=True)

    with col2:
        st.markdown('<div class="section-label">Detection result</div>',
                    unsafe_allow_html=True)
        if uploaded:
            img_np  = np.array(img)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            with st.spinner("Running inference..."):
                t0      = time.time()
                results = model.predict(
                    img_bgr, conf=conf_thresh, iou=iou_thresh,
                    verbose=False, show_labels=show_labels, show_conf=show_conf,
                )
                elapsed = (time.time() - t0) * 1000

            annotated     = results[0].plot(labels=show_labels, conf=show_conf, line_width=2)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_column_width=True)

            boxes = results[0].boxes
            st.markdown(f"""
            <div style="display:flex;gap:10px;margin:1rem 0;">
              <div class="metric-card" style="flex:1">
                <div class="label">Detections</div>
                <div class="value">{len(boxes)}</div>
              </div>
              <div class="metric-card" style="flex:1">
                <div class="label">Inference</div>
                <div class="value" style="font-size:20px">{elapsed:.0f}ms</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            if len(boxes) > 0:
                # ── Bar list ──
                st.markdown('<div class="section-label">Detected actions</div>',
                            unsafe_allow_html=True)
                sorted_boxes = sorted(boxes, key=lambda b: float(b.conf[0]), reverse=True)
                for box in sorted_boxes:
                    cls_id   = int(box.cls[0])
                    conf_v   = float(box.conf[0])
                    cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
                    bar_w    = int(conf_v * 100)
                    st.markdown(f"""
                    <div class="class-row">
                        <span class="cls-name">{cls_name}</span>
                        <div class="conf-bar-wrap">
                            <div class="conf-bar" style="width:{bar_w}%"></div>
                        </div>
                        <span class="conf-val">{conf_v:.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Chart ──
                st.markdown('<div class="section-label" style="margin-top:1.2rem">Confidence chart</div>',
                            unsafe_allow_html=True)
                names  = [CLASS_NAMES[int(b.cls[0])] if int(b.cls[0]) < len(CLASS_NAMES)
                          else f"class_{int(b.cls[0])}" for b in sorted_boxes]
                confs  = [float(b.conf[0]) for b in sorted_boxes]
                fig    = make_bar_chart(names, confs, "", "Confidence Score", "", dark)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No detections above confidence threshold")
        else:
            st.markdown("""
            <div style="height:300px;display:flex;align-items:center;
                        justify-content:center;border:1px dashed #1a2540;
                        border-radius:10px;color:#4a5a7a;font-size:14px;">
                Upload an image to see detections
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# TAB 2 — Video
# ============================================================
with tab2:
    col1, col2 = st.columns([1,1], gap="large")

    with col1:
        st.markdown('<div class="section-label">Input video</div>', unsafe_allow_html=True)
        video_file = st.file_uploader(
            "Upload surgical video", type=["mp4","avi","mov"],
            label_visibility="collapsed"
        )
        if video_file:
            st.video(video_file)

    with col2:
        st.markdown('<div class="section-label">Processed output</div>', unsafe_allow_html=True)

        if video_file:
            if st.button("▶  Run detection on video"):
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_in:
                    tmp_in.write(video_file.read())
                    tmp_in_path = tmp_in.name

                tmp_out_path = tmp_in_path.replace('.mp4', '_out.mp4')
                cap          = cv2.VideoCapture(tmp_in_path)
                w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps          = cap.get(cv2.CAP_PROP_FPS) or 25
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                out = cv2.VideoWriter(
                    tmp_out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
                )

                progress      = st.progress(0)
                status        = st.empty()
                frame_n       = 0
                action_counts = {}
                frame_nums    = []
                max_confs     = []

                while True:
                    ret, frame = cap.read()
                    if not ret: break

                    results   = model.predict(frame, conf=conf_thresh,
                                              iou=iou_thresh, verbose=False)
                    annotated = results[0].plot(labels=show_labels,
                                               conf=show_conf, line_width=2)
                    out.write(annotated)

                    boxes_f = results[0].boxes
                    max_c   = max([float(b.conf[0]) for b in boxes_f], default=0.0)
                    frame_nums.append(frame_n)
                    max_confs.append(max_c)

                    for box in boxes_f:
                        cls_id = int(box.cls[0])
                        name   = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
                        action_counts[name] = action_counts.get(name, 0) + 1

                    frame_n += 1
                    prog = min(frame_n / max(total_frames, 1), 1.0)
                    progress.progress(prog)
                    status.markdown(f"`Frame {frame_n}/{total_frames}`")

                cap.release()
                out.release()
                progress.progress(1.0)
                status.markdown("`Processing complete`")

                with open(tmp_out_path, 'rb') as f:
                    st.download_button(
                        "⬇  Download annotated video",
                        f, file_name="mesad_output.mp4", mime="video/mp4"
                    )

                if action_counts:
                    sorted_actions = sorted(action_counts.items(),
                                           key=lambda x: x[1], reverse=True)
                    top_actions    = sorted_actions[:10]

                    # ── Action frequency bar chart ──
                    st.markdown('<div class="section-label" style="margin-top:1rem">Action frequency</div>',
                                unsafe_allow_html=True)
                    fig_bar = make_bar_chart(
                        [a[0] for a in top_actions],
                        [a[1] for a in top_actions],
                        "Top 10 detected actions", "Frame count", "", dark
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                    # ── Pie chart ──
                    st.markdown('<div class="section-label">Action distribution</div>',
                                unsafe_allow_html=True)
                    fig_pie = make_pie_chart(
                        [a[0] for a in top_actions],
                        [a[1] for a in top_actions],
                        "Action share (top 10)", dark
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                    # ── Timeline ──
                    if frame_nums:
                        st.markdown('<div class="section-label">Confidence over time</div>',
                                    unsafe_allow_html=True)
                        fig_line = make_timeline_chart(
                            frame_nums, max_confs,
                            "Max detection confidence per frame", dark
                        )
                        st.plotly_chart(fig_line, use_container_width=True)

                os.unlink(tmp_in_path)

# ============================================================
# TAB 3 — Batch folder
# ============================================================
with tab3:
    st.markdown('<div class="section-label">Batch inference</div>', unsafe_allow_html=True)

    folder_path = st.text_input(
        "Path to folder of images",
        placeholder="e.g. C:/mesad/val/images"
    )
    max_images = st.slider("Max images to process", 5, 100, 20)

    if folder_path and st.button("▶  Run batch inference"):
        if not os.path.exists(folder_path):
            st.error(f"Folder not found: {folder_path}")
        else:
            images = [f for f in os.listdir(folder_path)
                      if f.lower().endswith(('.jpg','.jpeg','.png'))][:max_images]

            if not images:
                st.warning("No images found in folder")
            else:
                progress    = st.progress(0)
                results_list = []
                action_agg  = {}

                for i, img_name in enumerate(images):
                    img_path   = os.path.join(folder_path, img_name)
                    img_bgr    = cv2.imread(img_path)
                    results    = model.predict(img_bgr, conf=conf_thresh,
                                               iou=iou_thresh, verbose=False)
                    detections = []
                    for b in results[0].boxes:
                        cls_id = int(b.cls[0])
                        name   = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
                        conf_v = float(b.conf[0])
                        detections.append(f"{name} ({conf_v:.2f})")
                        action_agg[name] = action_agg.get(name, 0) + 1

                    results_list.append({
                        'Image'      : img_name,
                        'Detections' : len(results[0].boxes),
                        'Actions'    : ', '.join(detections) if detections else '—'
                    })
                    progress.progress((i+1) / len(images))

                progress.progress(1.0)
                st.success(f"✓ Processed {len(images)} images")

                import pandas as pd
                df = pd.DataFrame(results_list)

                # ── Summary metrics ──
                total_det  = df['Detections'].sum()
                avg_det    = df['Detections'].mean()
                images_hit = (df['Detections'] > 0).sum()

                mcol1, mcol2, mcol3 = st.columns(3)
                with mcol1:
                    st.markdown(f"""<div class="metric-card">
                        <div class="label">Total detections</div>
                        <div class="value">{total_det}</div>
                    </div>""", unsafe_allow_html=True)
                with mcol2:
                    st.markdown(f"""<div class="metric-card">
                        <div class="label">Avg per image</div>
                        <div class="value" style="font-size:22px">{avg_det:.1f}</div>
                    </div>""", unsafe_allow_html=True)
                with mcol3:
                    st.markdown(f"""<div class="metric-card">
                        <div class="label">Images w/ detections</div>
                        <div class="value">{images_hit}/{len(images)}</div>
                    </div>""", unsafe_allow_html=True)

                # ── Table — full width, scrollable ──
                st.markdown('<div class="section-label" style="margin-top:1rem">Results table</div>',
                            unsafe_allow_html=True)
                st.dataframe(
                    df,
                    use_container_width=True,
                    height=min(400, 38 * len(df) + 38),   # scrollable, capped at 400px
                    column_config={
                        "Image"      : st.column_config.TextColumn("Image", width="medium"),
                        "Detections" : st.column_config.NumberColumn("Detections", width="small",
                                                                      format="%d"),
                        "Actions"    : st.column_config.TextColumn("Actions", width="large"),
                    }
                )

                # ── Download CSV ──
                csv = df.to_csv(index=False)
                st.download_button(
                    "⬇  Download results CSV",
                    csv, file_name="mesad_batch_results.csv", mime="text/csv"
                )

                # ── Action frequency bar chart ──
                if action_agg:
                    st.markdown('<div class="section-label" style="margin-top:1rem">Action frequency across batch</div>',
                                unsafe_allow_html=True)
                    sorted_agg = sorted(action_agg.items(), key=lambda x: x[1], reverse=True)
                    fig_batch  = make_bar_chart(
                        [a[0] for a in sorted_agg],
                        [a[1] for a in sorted_agg],
                        f"All detected actions — {len(images)} images",
                        "Detection count", "", dark
                    )
                    st.plotly_chart(fig_batch, use_container_width=True)

                    # ── Pie chart ──
                    st.markdown('<div class="section-label">Action distribution</div>',
                                unsafe_allow_html=True)
                    fig_pie2 = make_pie_chart(
                        [a[0] for a in sorted_agg[:10]],
                        [a[1] for a in sorted_agg[:10]],
                        "Top 10 action share", dark
                    )
                    st.plotly_chart(fig_pie2, use_container_width=True)

                    # ── Detections per image bar ──
                    st.markdown('<div class="section-label">Detections per image</div>',
                                unsafe_allow_html=True)
                    fig_img = make_bar_chart(
                        list(df['Image']),
                        list(df['Detections']),
                        "Detection count by image", "Count", "", dark
                    )
                    st.plotly_chart(fig_img, use_container_width=True)
