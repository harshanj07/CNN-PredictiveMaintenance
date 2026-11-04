"""
Script to generate a 4-page professional technical whitepaper PDF:
Industry 4.0 Predictive Maintenance System
PyTorch 1D-CNN + Siamese Network Telemetry Architecture
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and print total page numbers (e.g. Page 1 of 4).
    Also prints clean running headers and footers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748B"))

        # Don't draw header on first page
        if self._pageNumber > 1:
            self.drawString(40, letter[1] - 26, "Industry 4.0 Predictive Maintenance — Technical Architecture & Model Guide")
            self.drawRightString(letter[0] - 40, letter[1] - 26, "PyTorch 1D-CNN + Siamese Network")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(40, letter[1] - 30, letter[0] - 40, letter[1] - 30)

        # Footer line and text
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(40, 30, letter[0] - 40, 30)

        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(letter[0] - 40, 20, page_str)
        self.drawString(40, 20, "Confidential & Proprietary — Engineering Project Documentation · Predictive Maintenance")
        self.restoreState()


def generate_pdf(output_path: str):
    # Printable area: width = 612 - 80 = 532 pt, height = 792 - 76 = 716 pt
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    primary_color = colors.HexColor("#0F2942")     # Deep Corporate Navy
    secondary_color = colors.HexColor("#0284C7")   # Precision Blue
    teal_accent = colors.HexColor("#0D9488")       # Industrial Teal
    dark_text = colors.HexColor("#1E293B")         # Charcoal Slate
    border_color = colors.HexColor("#CBD5E1")      # Light border
    bg_alt = colors.HexColor("#F8FAFC")            # Very light row bg

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=19,
        leading=22,
        textColor=primary_color,
        spaceAfter=2
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=teal_accent,
        spaceAfter=2
    )

    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.2,
        leading=9.5,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=5
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=primary_color,
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=secondary_color,
        spaceBefore=4,
        spaceAfter=2,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10.6,
        textColor=dark_text,
        spaceAfter=3.5
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10.6,
        textColor=dark_text,
        leftIndent=10,
        spaceAfter=2.8
    )

    tbl_header = ParagraphStyle(
        'TblHdr',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.2,
        leading=9,
        textColor=colors.white
    )

    tbl_cell = ParagraphStyle(
        'TblCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=6.8,
        leading=8.8,
        textColor=dark_text
    )

    tbl_cell_bold = ParagraphStyle(
        'TblCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=6.8,
        leading=8.8,
        textColor=primary_color
    )

    callout_box_style = ParagraphStyle(
        'CalloutBox',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.2,
        leading=10,
        textColor=colors.HexColor("#0F172A")
    )

    story = []

    # =========================================================================
    # PAGE 1: TITLE + EXEC SUMMARY + HOW 1D-CNN IS USED
    # =========================================================================
    story.append(Paragraph("Industry 4.0 Predictive Maintenance System", title_style))
    story.append(Paragraph("PyTorch 1D-CNN + Siamese Network Telemetry Platform · 18 Hydraulic Channels", subtitle_style))
    story.append(Paragraph("System Architecture · 1D-CNN Mechanics · 18-Sensor Catalog · Roadmap · Algorithms &amp; Models Inventory", meta_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=5, spaceBefore=0))

    # Section 1
    story.append(Paragraph("1. Executive Summary &amp; Project Purpose", h1_style))
    story.append(Paragraph(
        "<b>Project Overview:</b> In automated manufacturing, high-pressure hydraulic sub-assemblies (pumps, valves, actuators, and cylinders) "
        "operate under heavy load and continuous duty cycles. Unexpected breakdowns lead to catastrophic equipment failures, costly factory downtime, "
        "and operator hazards. This project implements a production-grade <b>Predictive Maintenance (PdM) platform</b> that continuously monitors "
        "multi-channel sensor telemetry, models temporal signatures, detects degradation patterns early, and flags equipment anomalies "
        "<i>before</i> functional breakdown occurs.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Dual Deep Learning Paradigm:</b> Industrial telemetry features both known repetitive faults and unpredictable anomalies. "
        "The project pairs two complementary PyTorch networks: (1) a <b>1D Convolutional Neural Network (1D-CNN)</b> that performs high-speed "
        "binary fault classification (Normal: 0 vs. Fault: 1), and (2) a <b>Siamese Metric-Learning Network</b> that projects operational cycles "
        "into a 128-dimensional embedding space, calculating pairwise Euclidean distance against healthy baseline cycles for few-shot anomaly detection.",
        body_style
    ))

    # Section 2
    story.append(Paragraph("2. Deep Dive: How the 1D-CNN is Used in the Project", h1_style))
    story.append(Paragraph(
        "The 1D-CNN (implemented in <font face='Courier' color='#0F2942'>src/cnn_model.py</font>) serves as the primary real-time fault detector. "
        "Unlike standard 2D-CNNs designed for spatial image grids, <b>1D-CNNs slide convolutional filters across time steps</b>, capturing temporal "
        "dynamics, rate-of-change, and inter-sensor correlations across all 18 hydraulic telemetry channels simultaneously.",
        body_style
    ))

    story.append(Paragraph("<b>Input Representation &amp; Temporal Windowing:</b>", h2_style))
    story.append(Paragraph(
        "• <b>Tensor Shape:</b> The network operates on temporal sliding windows of shape <code>(Batch Size, Sequence Length = 8, Features = 18)</code>.<br/>"
        "• <b>Channel Transposition:</b> In <code>forward(x)</code>, the tensor is permuted to <code>(Batch Size, Channels = 18, Time = 8)</code>. "
        "The 18 hydraulic sensors act as input channels and 8 consecutive time ticks form the temporal receptive field.<br/>"
        "• <b>Single-Tick Tiling for Inference:</b> When individual sensor readings arrive via the FastAPI or Streamlit UI, the reading is tiled "
        "across 8 timesteps to form a stationary synthetic window <code>(1, 8, 18)</code>, guaranteeing identical batch and streaming inference pipelines.",
        body_style
    ))

    story.append(Paragraph("<b>Detailed Neural Architecture:</b>", h2_style))
    story.append(Paragraph(
        "<b>1. Three Hierarchical 1D-Convolutional Feature Blocks:</b><br/>"
        "&nbsp;&nbsp;• <i>Block 1:</i> <code>Conv1d(in=18, out=32, kernel=3, padding=1)</code> → <code>BatchNorm1d(32)</code> → <code>ReLU()</code> → <code>MaxPool1d(2, stride=2)</code> [Time: 8 → 4]<br/>"
        "&nbsp;&nbsp;• <i>Block 2:</i> <code>Conv1d(in=32, out=64, kernel=3, padding=1)</code> → <code>BatchNorm1d(64)</code> → <code>ReLU()</code> → <code>MaxPool1d(2, stride=2)</code> [Time: 4 → 2]<br/>"
        "&nbsp;&nbsp;• <i>Block 3:</i> <code>Conv1d(in=64, out=128, kernel=3, padding=1)</code> → <code>BatchNorm1d(128)</code> → <code>ReLU()</code> → <code>MaxPool1d(2, stride=2)</code> [Time: 2 → 1]<br/>"
        "<i>Engineering Role:</i> Kernels automatically learn filter weights that detect physical phenomena: pressure surges, current spikes, "
        "phase delays between displacement and power, and solenoid voltage drops. BatchNorm stabilizes activation distributions, and MaxPool1d reduces temporal noise.",
        body_style
    ))

    story.append(Paragraph(
        "<b>2. Fully Connected Dense Classification Head:</b><br/>"
        "&nbsp;&nbsp;• <i>Flattening:</i> Collapses feature maps into a 128-dimensional representation (128 channels × 1 timestep).<br/>"
        "&nbsp;&nbsp;• <i>Dense Layers:</i> <code>Linear(128, 256)</code> → <code>ReLU()</code> → <code>Dropout(0.5)</code> → <code>Linear(256, 128)</code> → <code>ReLU()</code> → <code>Dropout(0.5)</code> → <code>Linear(128, 2)</code>.<br/>"
        "<i>Engineering Role:</i> 50% Dropout mitigates co-adaptation and prevents overfitting on specific operating cycles. Output layer generates raw 2-class logits.",
        body_style
    ))

    story.append(Paragraph(
        "<b>3. Loss Optimization &amp; Inference Confidence:</b><br/>"
        "The model is trained via <code>nn.CrossEntropyLoss</code> using the <code>Adam</code> optimizer (learning rate = 0.001) with early stopping. "
        "In production inference, <code>Softmax</code> transforms output logits into normalized class probabilities: Class 0 denotes <b>Normal</b>, "
        "and Class 1 denotes <b>Fault</b>. The highest probability is surfaced as the confidence metric.",
        body_style
    ))

    # =========================================================================
    # PAGE 2: SENSOR FEATURES CATALOG (18 FEATURES TABLE)
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("3. Sensor Features Catalog: 18 Telemetry Channels Explained", h1_style))
    story.append(Paragraph(
        "The industrial hydraulic system monitors 6 sub-assemblies (BLO, BHL, BHR, BRU, HR, HL). Each module captures 3 signals: "
        "<b>Weg</b> (actuator displacement current in mA), <b>Power</b> (active wattage in W), and <b>Voltage</b> (excitation potential in V). "
        "Below is the complete feature dictionary detailing why each sensor channel is monitored:",
        body_style
    ))

    feature_table_data = [
        [
            Paragraph("<b>Feature Column Name</b>", tbl_header),
            Paragraph("<b>Sub-Assembly &amp; Signal Type</b>", tbl_header),
            Paragraph("<b>Why It Is Used (Engineering Explanation)</b>", tbl_header)
        ],
        [
            Paragraph("<code>I_w_BLO_Weg</code>", tbl_cell_bold),
            Paragraph("Bottom Left Outlet · Displacement", tbl_cell),
            Paragraph("Measures actuator stroke current to detect mechanical travel delays and sticking in the bottom-left outlet valve.", tbl_cell)
        ],
        [
            Paragraph("<code>O_w_BLO_power</code>", tbl_cell_bold),
            Paragraph("Bottom Left Outlet · Power", tbl_cell),
            Paragraph("Monitors electrical power draw to detect excess motor workload, mechanical resistance, or valve binding.", tbl_cell)
        ],
        [
            Paragraph("<code>O_w_BLO_voltage</code>", tbl_cell_bold),
            Paragraph("Bottom Left Outlet · Voltage", tbl_cell),
            Paragraph("Tracks operating supply voltage to identify line voltage drops, power instability, or solenoid coil degradation.", tbl_cell)
        ],
        [
            Paragraph("<code>I_w_BHL_Weg</code>", tbl_cell_bold),
            Paragraph("Bottom Left Hydraulic · Displacement", tbl_cell),
            Paragraph("Measures cylinder piston position current to detect stick-slip motion or uneven mechanical stroke travel.", tbl_cell)
        ],
        [
            Paragraph("<code>O_w_BHL_power</code>", tbl_cell_bold),
            Paragraph("Bottom Left Hydraulic · Power", tbl_cell),
            Paragraph("Measures active power consumption to identify pressure drops, internal fluid bypass, or hydraulic pump wear.", tbl_cell)
        ],
        [
            Paragraph("<code>O_w_BHL_voltage</code>", tbl_cell_bold),
            Paragraph("Bottom Left Hydraulic · Voltage", tbl_cell),
            Paragraph("Monitors actuator control voltage to catch solenoid switching lag and control circuit voltage degradation.", tbl_cell)
        ],
        [
            Paragraph("<code>I_w_BHR_Weg</code>", tbl_cell_bold),
            Paragraph("Bottom Right Hydraulic · Displacement", tbl_cell),
            Paragraph("Tracks linear stroke displacement current to verify synchronized cylinder movement with the left hydraulic bank.", tbl_cell)
        ],
        [
            Paragraph("<code>O_w_BHR_power</code>", tbl_cell_bold),
            Paragraph("Bottom Right Hydraulic · Power", tbl_cell),
            Paragraph("Monitors active power draw to spot excessive mechanical resistance, hydraulic fluid throttling, or blockages.", tbl_cell)
        ],
        [
            Paragraph("<code>O_w_BHR_voltage</code>", tbl_cell_bold),
            Paragraph("Bottom Right Hydraulic · Voltage", tbl_cell),
            Paragraph("Captures terminal voltage levels to identify loose wiring connections, contact resistance, or supply dips.", tbl_cell)
        ],
        [
            Paragraph("<code>I_w_BRU_Weg</code>", tbl_cell_bold),
            Paragraph("Bottom Right Upper · Displacement", tbl_cell),
            Paragraph("Tracks upper actuator displacement current to verify proper valve seat engagement and stroke travel limits.", tbl_cell)
        ],
        [
            Paragraph("<code>O_w_BRU_power</code>", tbl_cell_bold),
            Paragraph("Bottom Right Upper · Power", tbl_cell),
            Paragraph("Quantifies power draw to detect motor overload, fluid contamination drag, or hydraulic cylinder seal friction.", tbl_cell)
        ],
        [
            Paragraph("<code>O_w_BRU_voltage</code>", tbl_cell_bold),
            Paragraph("Bottom Right Upper · Voltage", tbl_cell),
            Paragraph("Records drive voltage across the upper actuator to verify reliable solenoid energization across cycles.", tbl_cell)
        ],
        [
            Paragraph("<code>I_w_HR_Weg</code>", tbl_cell_bold),
            Paragraph("Hydraulic Right · Displacement", tbl_cell),
            Paragraph("Measures main right cylinder displacement current to verify structural alignment and full hydraulic stroke cycle.", tbl_cell)
        ],
        [
            Paragraph("<code>O_w_HR_power</code>", tbl_cell_bold),
            Paragraph("Hydraulic Right · Power", tbl_cell),
            Paragraph("Tracks main right actuator power consumption to identify relief valve leaks, pump slip, or pressure loss.", tbl_cell)
        ],
        [
            Paragraph("<code>O_w_HR_voltage</code>", tbl_cell_bold),
            Paragraph("Hydraulic Right · Voltage", tbl_cell),
            Paragraph("Monitors main right valve operating voltage to detect power fluctuations and solenoid thermal degradation.", tbl_cell)
        ],
        [
            Paragraph("<code>I_w_HL_Weg</code>", tbl_cell_bold),
            Paragraph("Hydraulic Left · Displacement", tbl_cell),
            Paragraph("Measures main left cylinder displacement current to detect mechanical misalignment and cylinder seal wear.", tbl_cell)
        ],
        [
            Paragraph("<code>O_w_HL_power</code>", tbl_cell_bold),
            Paragraph("Hydraulic Left · Power", tbl_cell),
            Paragraph("Measures left actuator wattage to detect hydraulic cavitation, mechanical binding, or fluid viscosity changes.", tbl_cell)
        ],
        [
            Paragraph("<code>O_w_HL_voltage</code>", tbl_cell_bold),
            Paragraph("Hydraulic Left · Voltage", tbl_cell),
            Paragraph("Tracks left hydraulic valve supply voltage to identify electrical brownouts and control line degradation.", tbl_cell)
        ],
    ]

    col_widths = [110, 130, 292]
    t_feat = Table(feature_table_data, colWidths=col_widths, repeatRows=1)
    t_feat.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 3.5),
        ('TOPPADDING', (0, 0), (-1, 0), 3.5),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 2.2),
        ('TOPPADDING', (0, 1), (-1, -1), 2.2),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_alt]),
    ]))
    story.append(t_feat)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<i>Dataset Meta-Attributes:</i> <code>Timestamp</code> (cyclical index partitioning continuous high-frequency measurements into operational duty cycles) "
        "and <code>Labels</code> (ground truth binary target: 0 = Normal / Healthy, 1 = Fault / Anomaly).",
        meta_style
    ))

    # =========================================================================
    # PAGE 3: INDUSTRIAL APPLICATIONS & FUTURE ENHANCEMENTS
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("4. Industrial Value &amp; Target Deployment Sectors", h1_style))
    story.append(Paragraph(
        "Predictive maintenance delivers immediate operational ROI by transforming plant strategy from reactive repair to Condition-Based Maintenance (CBM). "
        "Below are 6 primary industrial domains where this platform can be directly deployed:",
        body_style
    ))

    use_cases = [
        ("Automotive Stamping &amp; Press Lines",
         "Heavy mechanical-hydraulic stamping presses exert 500+ tons per stroke. Detecting cylinder displacement lag and valve power spikes early prevents costly die damage, stamping defects, and sudden press jamming."),
        ("Industrial Injection Molding &amp; Metal Extrusion",
         "Continuous metal extrusion machines require synchronized bilateral hydraulic force. Monitoring power and displacement balance between left and right modules flags piston seal wear and fluid thermal breakdown before seal blowouts occur."),
        ("Aerospace Flight Control Rigs &amp; Ground Test Benches",
         "Flight control hydraulic actuators require zero-failure reliability. The Siamese network's similarity metric flags subtle mechanical anomalies in test rigs weeks before physical failure occurs."),
        ("Oil, Gas &amp; Offshore Subsea Hydraulic Units (HPUs)",
         "Hydraulic units operating subsea Blowout Preventers (BOPs) operate in harsh, remote environments where unplanned downtime costs over $1M/day. Early warning of solenoid valve drop allows planned offshore servicing."),
        ("Heavy Construction &amp; Mining Mobile Machinery",
         "Excavators and tunnel boring machines face extreme shock and dirt contamination. Real-time telemetry monitoring prevents catastrophic main hydraulic pump cavitation and hydraulic cylinder scoring."),
        ("Renewable Energy (Hydroelectric &amp; Wind Turbine Governors)",
         "Hydraulic governors modulate water flow in hydro turbines, and pitch cylinders angle wind turbine blades. Automated fault monitoring prevents dangerous turbine runaway events and maximizes continuous energy generation.")
    ]
    for title, desc in use_cases:
        story.append(Paragraph(f"• <b>{title}:</b> {desc}", bullet_style))

    story.append(Spacer(1, 4))
    story.append(Paragraph("5. Future Enhancements &amp; Technological Roadmap", h1_style))
    story.append(Paragraph(
        "The current codebase establishes an end-to-end operational pipeline. To scale into enterprise industrial deployments, "
        "the following technological enhancements represent the strategic product roadmap:",
        body_style
    ))

    roadmap = [
        ("Multi-Class &amp; Hierarchical Fault Diagnosis",
         "Extend binary classification (0 vs 1) into granular multi-class fault categorization: classifying specific sub-assembly failure modes (e.g., valve stick-slip, internal fluid bypass, pump cavitation, cylinder seal scoring, accumulator pressure discharge)."),
        ("Remaining Useful Life (RUL) Prognostics",
         "Incorporate temporal regression heads (Temporal Convolutional Networks or Survival Analysis) that output the estimated operating hours or cycles remaining until functional failure, empowering automated replacement scheduling."),
        ("Edge AI Model Quantization (INT8 / TensorRT / ONNX)",
         "Quantize models to INT8 precision and export to ONNX / TensorRT for execution on embedded industrial edge hardware (NVIDIA Jetson, Siemens IPCs, Raspberry Pi 5) with sub-millisecond local inference and no cloud dependency."),
        ("Multi-Modal Sensor Fusion (Acoustic &amp; Vibration Telemetry)",
         "Integrate vibration accelerometers and high-frequency acoustic emission signals (as researched in <code>notebooks/PM-CRNN.ipynb</code>) using cross-attention transformers to detect micro-cracks before electrical/hydraulic deviations emerge."),
        ("Self-Supervised Industrial Foundation Models",
         "Pre-train masked autoencoder time-series transformers (PatchTST, TimesNet, or Chronos) on massive unlabeled industrial fleet telemetry, enabling fine-tuning on new production lines with minimal labeled historical failure data."),
        ("Generative AI Maintenance Copilot &amp; Automated Work Orders",
         "Feed SHAP feature attributions into a domain-specific LLM agent to automatically generate plain-language diagnostic tickets, root-cause explanations, and recommended replacement part orders for shop-floor technicians."),
        ("Industrial IoT Protocols &amp; Closed-Loop SCADA Integration",
         "Implement native OPC-UA and MQTT connectors to interface directly with factory PLCs (Siemens S7, Allen-Bradley), allowing automated closed-loop safety throttling or graceful machine shutdown when fault risk crosses critical thresholds.")
    ]
    for title, desc in roadmap:
        story.append(Paragraph(f"• <b>{title}:</b> {desc}", bullet_style))

    # =========================================================================
    # PAGE 4: COMPREHENSIVE CATALOG OF MODELS & ALGORITHMS WITH FILE LOCATIONS
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("6. Catalog of Models &amp; Algorithms Across Codebase", h1_style))
    story.append(Paragraph(
        "Below is the complete inventory of all neural network models, statistical algorithms, preprocessing routines, "
        "and deployment frameworks utilized across both the core production system (<font face='Courier'>src/</font>) "
        "and exploratory research notebooks (<font face='Courier'>notebooks/</font>):",
        body_style
    ))

    algo_table_data = [
        [
            Paragraph("<b>Model / Algorithm</b>", tbl_header),
            Paragraph("<b>Category</b>", tbl_header),
            Paragraph("<b>File Location</b>", tbl_header),
            Paragraph("<b>Role &amp; Functionality in Project</b>", tbl_header)
        ],
        [
            Paragraph("<b>1D-CNN (Convolutional Neural Net)</b>", tbl_cell_bold),
            Paragraph("Deep Learning", tbl_cell),
            Paragraph("<code>src/cnn_model.py</code>", tbl_cell),
            Paragraph("Primary supervised classifier with 3 Conv1d blocks + 3 dense layers; detects temporal fault signatures.", tbl_cell)
        ],
        [
            Paragraph("<b>Siamese Neural Network</b>", tbl_cell_bold),
            Paragraph("Metric Learning", tbl_cell),
            Paragraph("<code>src/siamese_model.py</code>", tbl_cell),
            Paragraph("Twin Conv1D encoders mapping telemetry into 128-d metric space; evaluates Euclidean distance for anomaly similarity.", tbl_cell)
        ],
        [
            Paragraph("<b>CRNN (Conv1D + Bidirectional GRU)</b>", tbl_cell_bold),
            Paragraph("Hybrid DL / Research", tbl_cell),
            Paragraph("<code>notebooks/PM-CRNN.ipynb</code>", tbl_cell),
            Paragraph("Explores spatio-temporal modeling combining 1D convolutions with recurrent gating (GRU) and audio/frequency spectrograms.", tbl_cell)
        ],
        [
            Paragraph("<b>1D-ResNet (Residual CNN)</b>", tbl_cell_bold),
            Paragraph("Deep Learning / Research", tbl_cell),
            Paragraph("<code>notebooks/PMResnet.ipynb</code>", tbl_cell),
            Paragraph("Experiments with deep residual skip connections (<code>residual_block</code>) to prevent gradient degradation on long sequences.", tbl_cell)
        ],
        [
            Paragraph("<b>SHAP (KernelExplainer)</b>", tbl_cell_bold),
            Paragraph("Explainable AI (XAI)", tbl_cell),
            Paragraph("<code>src/feature_analysis.py</code>", tbl_cell),
            Paragraph("Game-theoretic feature attribution calculating SHAP values across time steps to rank individual sensor importance.", tbl_cell)
        ],
        [
            Paragraph("<b>t-SNE (t-Distributed Stochastic Neighbor)</b>", tbl_cell_bold),
            Paragraph("Dimensionality Reduction", tbl_cell),
            Paragraph("<code>src/visualization.py</code><br/><code>src/train.py</code>", tbl_cell),
            Paragraph("Projects high-dimensional latent layer activations into 2D plots to visualize separation between healthy and faulty states.", tbl_cell)
        ],
        [
            Paragraph("<b>StandardScaler &amp; Sequence Windowing</b>", tbl_cell_bold),
            Paragraph("Data Preprocessing", tbl_cell),
            Paragraph("<code>src/data_processor.py</code>", tbl_cell),
            Paragraph("Applies Z-score scaling and sliding window segmentation to convert continuous time logs into <code>(N, 8, 18)</code> sequences.", tbl_cell)
        ],
        [
            Paragraph("<b>Pairwise Siamese Generator</b>", tbl_cell_bold),
            Paragraph("Dataset Algorithm", tbl_cell),
            Paragraph("<code>src/data_processor.py</code>", tbl_cell),
            Paragraph("Generates balanced positive pairs (label 1: same state) and negative pairs (label 0: contrasting state) for contrastive training.", tbl_cell)
        ],
        [
            Paragraph("<b>ROC-AUC &amp; Confusion Matrix</b>", tbl_cell_bold),
            Paragraph("Statistical Evaluation", tbl_cell),
            Paragraph("<code>src/evaluation.py</code><br/><code>src/cnn_model.py</code>", tbl_cell),
            Paragraph("Calculates classification accuracy, precision, recall, F1-scores, decision threshold curves, and confusion heatmaps.", tbl_cell)
        ],
        [
            Paragraph("<b>FastAPI Async Inference Server</b>", tbl_cell_bold),
            Paragraph("Backend API", tbl_cell),
            Paragraph("<code>src/api/app.py</code>", tbl_cell),
            Paragraph("Provides high-performance RESTful microservice with lifespan model loading, health checks, and JSON <code>/predict</code> endpoints.", tbl_cell)
        ],
        [
            Paragraph("<b>Streamlit Reactive Web App</b>", tbl_cell_bold),
            Paragraph("User Interface", tbl_cell),
            Paragraph("<code>app.py</code>", tbl_cell),
            Paragraph("Single-file interactive UI featuring single-sample manual input, sliding-window CSV upload, and model evaluation plot gallery.", tbl_cell)
        ],
    ]

    col_widths_algo = [115, 80, 115, 222]
    t_algo = Table(algo_table_data, colWidths=col_widths_algo, repeatRows=1)
    t_algo.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 3.5),
        ('TOPPADDING', (0, 0), (-1, 0), 3.5),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 2.2),
        ('TOPPADDING', (0, 1), (-1, -1), 2.2),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_alt]),
    ]))
    story.append(t_algo)

    story.append(Spacer(1, 6))
    # Summary callout block
    callout_data = [[
        Paragraph(
            "<b>Architectural Takeaway:</b> The repository demonstrates a complete engineering evolution: "
            "from exploratory research in Jupyter notebooks (evaluating CRNNs, ResNets, and spectrograms) to a lean, "
            "production-grade PyTorch implementation. The core 1D-CNN delivers rapid, deterministic fault classification, "
            "while the Siamese Network provides robust baseline-comparison anomaly scoring, backed by full SHAP explainability, "
            "FastAPI microservices, and an interactive Streamlit UI.",
            callout_box_style
        )
    ]]
    t_callout = Table(callout_data, colWidths=[532])
    t_callout.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#3B82F6")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_callout)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF built successfully: {output_path}")

if __name__ == "__main__":
    out_file = Path("d:/projects/CNN-PredictiveMaintenance/CNN-PredictiveMaintenance/Predictive_Maintenance_Project_Documentation.pdf")
    generate_pdf(str(out_file))
