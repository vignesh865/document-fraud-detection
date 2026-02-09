``` mermaid
graph TD
    %% --- Initial Setup ---
    Start(Start: Digital Document Intake) --> Input[Input Document File]
    Input --> Triage{Stage 1: Initial Triage & Metadata}

    %% --- STAGE 1: Metadata & Header Analysis ---
    subgraph Stage1 [Stage 1: Triage & Metadata Forensics]
        Triage -- "Identify File Type (Magic Bytes)" --> FileID[File Type Identification]
        FileID --> MetaExt[Extract EXIF/XMP Metadata]
        MetaExt --> MetaCheck{Metadata Flags?}
        MetaCheck -- "Suspicious Software (e.g., Photoshop, Canva)" --> FlagMeta[FLAG: Software Signature]
        MetaCheck -- "Timeline Paradox (Created > Modified)" --> FlagTime[FLAG: Timestamp Inconsistency]
        MetaCheck -- "Hardware Mismatch (e.g., Scanner type)" --> FlagHW[FLAG: Hardware Profile Mismatch]
        FlagMeta --> Routing
        FlagTime --> Routing
        FlagHW --> Routing
        MetaCheck -- "Clean Metadata" --> Routing
    end

    %% --- Routing Decision ---
    Routing{File Classification Route}

    %% --- STAGE 2: Special Logical Checks (IDs/Passports) ---
    Routing -- "Document contains MRZ (Passport/ID)" --> MRZCheck{Stage 2: Logical Validation}
    subgraph Stage2 [Stage 2: MRZ & Logical Checks]
        MRZCheck --> ExtractMRZ[Extract Machine Readable Zone Text]
        ExtractMRZ --> RunChecksum[Execute Modulo-10 Checksum Algorithm]
        RunChecksum --> ValidMRZ{Checksum Valid?}
        ValidMRZ -- No --> FlagMRZ[CRITICAL FLAG: Mathematical Impossibility]
        ValidMRZ -- Yes --> PassMRZ[Pass: MRZ Logic Valid]
    end
    PassMRZ --> MainBranch
    FlagMRZ --> ScoreAgg

    %% --- Main Branching based on Container Format ---
    MainBranch{Container Format?}

    %% --- STAGE 3: PDF Container Forensics ---
    MainBranch -- "PDF (Native or Scanned)" --> PDFAnalysis{Stage 3: PDF Structure}
    subgraph Stage3 [Stage 3: PDF Container Analysis]
        PDFAnalysis --> IncUpdate[Check Incremental Updates / Appended Data]
        IncUpdate -- "Found hidden previous versions" --> FlagInc[FLAG: Hidden History Found]
        PDFAnalysis --> GhostLayer[Analyze Layers & Opacity]
        GhostLayer -- "Hidden Objects (0% Opacity)" --> FlagGhost[FLAG: Ghost Objects Detected]
        PDFAnalysis --> ObjExtract[Extract Image Objects & Resources]
        ObjExtract -- "Multiple detached image objects on one page" --> FlagPaste[FLAG: Cut-and-Paste Objects]
        ObjExtract --> ExtractedImages(Result: Raw Extracted Images)
    end
    FlagInc --> ScoreAgg
    FlagGhost --> ScoreAgg
    FlagPaste --> ScoreAgg
    ExtractedImages --> ImageBranch

    %% --- Path for direct Image files ---
    MainBranch -- "Direct Image (JPG, PNG, BMP, TIFF)" --> ImageBranch

    %% --- STAGE 4: Image Specific Forensics ---
    ImageBranch{Image Compression Type?}

    %% --- STAGE 4a: Lossy (JPEG) ---
    subgraph Stage4a [Stage 4a: Lossy / JPEG Forensics]
        ImageBranch -- "Lossy (JPEG)" --> ELA[Execute Error Level Analysis - ELA]
        ELA -- "High contrast non-uniform blocks" --> FlagELA[FLAG: Compression Inconsistency]
        ImageBranch -- "Lossy (JPEG)" --> GhostJPEG[Detect JPEG Ghosting / Double Compression]
        GhostJPEG -- "Multiple quantization tables detected" --> FlagDoubleSave[FLAG: Re-saved Image]
    end
    FlagELA --> UniversalVisual
    FlagDoubleSave --> UniversalVisual
    ELA -- "Uniform noise" --> UniversalVisual

    %% --- STAGE 4b: Lossless (PNG/BMP/TIFF) ---
    subgraph Stage4b [Stage 4b: Lossless / Raw Forensics]
        ImageBranch -- "Lossless/Raw (PNG, BMP, TIFF)" --> NoiseMap[Analyze Sensor Noise Variance]
        NoiseMap -- "Broken noise pattern in specific region" --> FlagNoise[FLAG: Localized Manipulation]
        ImageBranch -- "Lossless/Raw (PNG, BMP, TIFF)" --> BPCS[Bit-Plane Complexity Segmentation]
        BPCS -- "Unnatural lack of randomness in LSB" --> FlagBPCS[FLAG: Bit-Plane Anomaly]
    end
    FlagNoise --> UniversalVisual
    FlagBPCS --> UniversalVisual
    NoiseMap -- "Consistent noise grain" --> UniversalVisual

    %% --- STAGE 5: Universal Visual & AI Analysis ---
    subgraph Stage5 [Stage 5: Universal Visual & Advanced AI]
        UniversalVisual(Merge Image Paths) --> CopyMove[Run Copy-Move Detection - SIFT/SURF]
        CopyMove -- "Identical pixel clones found" --> FlagClone[FLAG: Cloning/Duplication detected]
        
        UniversalVisual --> SkewGrid[Analyze Text Skew & Alignment Grid]
        SkewGrid -- "Inconsistent text angles/alignment" --> FlagSkew[FLAG: Insertion / Alignment Error]

        UniversalVisual --> OCRCheck[Perform OCR & Compare to Visual]
        OCRCheck -- "Text layer mismatch vs Visual pixels" --> FlagOCR[FLAG: Data Inconsistency]

        UniversalVisual --> AdvAI{AI & Specialized Models}
        AdvAI --> MoireDet[Moiré Pattern Detection]
        MoireDet -- "Periodic grid interference patterns" --> FlagRecap[FLAG: Recapture / Screen Photo]
        AdvAI --> GANDet[GAN / Deepfake Artifact Scanner]
        GANDet -- "Synthetic texture/glitches detected" --> FlagGAN[FLAG: AI-Generated Content]
    end
    FlagClone --> ScoreAgg
    FlagSkew --> ScoreAgg
    FlagOCR --> ScoreAgg
    FlagRecap --> ScoreAgg
    FlagGAN --> ScoreAgg

    %% --- STAGE 6: Scoring & Verdict ---
    subgraph Stage6 [Stage 6: Synthesis & Verdict]
        ScoreAgg(Aggregate All Flags & Weights) --> RiskScore{Calculate Risk Score}
        RiskScore -- "Score > Threshold High" --> HighRisk[Verdict: HIGH RISK / FRAUD]
        RiskScore -- "Score in Medium Range" --> ManualRev[Verdict: SUSPICIOUS / MANUAL REVIEW]
        RiskScore -- "Score < Threshold Low" --> Authentic[Verdict: LIKELY AUTHENTIC]
    end

    HighRisk --> End((End Process))
    ManualRev --> End((End Process))
    Authentic --> End((End Process))

    %% Styling
    classDef startEnd fill:#f9f,stroke:#333,stroke-width:2px;
    classDef process fill:#e1f5fe,stroke:#0277bd,stroke-width:1px;
    classDef decision fill:#fff9c4,stroke:#fbc02d,stroke-width:1px;
    classDef flag fill:#ffcdd2,stroke:#c62828,stroke-width:2px,color:#c62828;
    classDef subgraphStyle fill:#f5f5f5,stroke:#616161,stroke-width:1px,stroke-dasharray: 5 5;

    class Start,End,Input,ExtractedImages startEnd;
    class FileID,MetaExt,ExtractMRZ,RunChecksum,IncUpdate,GhostLayer,ObjExtract,ELA,GhostJPEG,NoiseMap,BPCS,CopyMove,SkewGrid,OCRCheck,MoireDet,GANDet,ScoreAgg process;
    class Triage,MetaCheck,Routing,MRZCheck,ValidMRZ,MainBranch,PDFAnalysis,ImageBranch,UniversalVisual,AdvAI,RiskScore decision;
    class FlagMeta,FlagTime,FlagHW,FlagMRZ,FlagInc,FlagGhost,FlagPaste,FlagELA,FlagDoubleSave,FlagNoise,FlagBPCS,FlagClone,FlagSkew,FlagOCR,FlagRecap,FlagGAN,HighRisk flag;
    class Stage1,Stage2,Stage3,Stage4a,Stage4b,Stage5,Stage6 subgraphStyle;
```