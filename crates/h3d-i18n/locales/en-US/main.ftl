# HISTO3D - English (en-US), secondary language.
# Keys must match id-ID/main.ftl exactly (checked by the key_parity test).
# ASCII punctuation only; the non-ASCII characters used (µ ² ³) are checked by
# the ui_glyph_coverage test in h3d-ui.

# --- App and About ---
app-name = HISTO3D
app-title = { $name } - HISTO3D
app-tagline = 3D tumor reconstruction from serial histology slides
about-title = About HISTO3D
about-research-only = Hanya untuk riset, bukan untuk diagnosis
about-version = Version { $version }
about-license = MIT or Apache-2.0 license

# --- Language and theme ---
lang-current = Language
lang-id = Bahasa Indonesia
lang-en = English
theme-light = Light
theme-dark = Dark

# --- App modes ---
mode-slides = Slides
mode-register = Register
mode-segment = Segment
mode-reconstruct = Reconstruct
mode-report = Report

# --- Common actions ---
action-new = New project
action-open = Open
action-save = Save
action-save-as = Save as
action-import = Import
action-export = Export
action-accept = Accept
action-reject = Reject
action-undo = Undo
action-redo = Redo
action-cancel = Cancel
action-apply = Apply
action-close = Close
action-ok = OK
action-delete = Delete
action-add = Add
action-edit = Edit
action-reset = Reset
action-run = Run
action-stop = Stop
action-help = Help
action-settings = Settings
action-prev-slice = Previous slice
action-next-slice = Next slice
action-zoom-in = Zoom in
action-zoom-out = Zoom out
action-zoom-fit = Fit to screen

# --- Common ---
common-yes = Yes
common-no = No
common-none = None
common-unknown = Unknown
common-loading = Loading...
common-working = Working...
common-done = Done
common-slice-n = Slice { $n }
common-slices-count = { $n } slices
common-value-with-uncertainty = { $value } +/- { $uncertainty } { $unit }
unit-um = µm
unit-mm = mm
unit-mm2 = mm²
unit-mm3 = mm³
unit-percent = %

# --- Slides mode ---
slides-import = Import slides
slides-import-hint = Choose slide files (SVS, TIFF, NDPI, PNG, JPEG)
slides-grid = Thumbnail grid
slides-mpp = Micrometers per pixel (mpp)
slides-mpp-x = mpp X
slides-mpp-y = mpp Y
mpp-required = mpp not found in metadata; enter the mpp value manually (it is never guessed)
slides-spacing = Slice spacing (µm)
slides-z-position = Z position (µm)
slides-order = Order
slides-order-source-filename = From file name
slides-order-source-manual = Set manually
slides-order-source-suggested = Accepted suggestion
slides-drag-to-reorder = Drag to reorder
slides-mark-damaged = Mark damaged
slides-unmark-damaged = Unmark damaged
slides-exclude = Exclude
slides-include = Include
slides-status-ok = In use
slides-status-excluded = Excluded
slides-status-damaged = Damaged
slides-stain = Stain
slides-stain-he = H&E
slides-stain-ihc = IHC
slides-stain-other = Other
slides-suggest-order = Suggest order and orientation
slides-suggest-preview = Suggestion preview
slides-flip-order = Flip order
slides-uncertain-slice = Uncertain slice
slides-uncertain-hint = The suggester marked this slice as uncertain; check it before accepting
slides-orientation = Orientation
slides-orientation-flip-h = Flip horizontally
slides-orientation-flip-v = Flip vertically
slides-orientation-rotate = Rotate 90 degrees
slides-file-name = File name
slides-size-px = Size (pixels)
slides-size-um = Size (µm)
slides-sample-name = Sample name
slides-sample-notes = Notes
slides-shrinkage = Linear shrinkage factor

# --- Clinical panel (ClinicalMeta) ---
clinical-panel = Clinical
clinical-hint = Pseudonymous data; stored separately in clinical.json and never used as an input to any computation
clinical-subject-pseudo-id = Subject pseudonym ID
clinical-age-years = Age (years)
clinical-sex = Sex
clinical-sex-female = Female
clinical-sex-male = Male
clinical-sex-other = Other
clinical-sex-unknown = Unknown
clinical-topography-icdo = ICD-O topography
clinical-morphology-icdo = ICD-O morphology
clinical-laterality = Laterality
clinical-laterality-left = Left
clinical-laterality-right = Right
clinical-laterality-bilateral = Bilateral
clinical-laterality-not-applicable = Not applicable
clinical-procedure = Procedure
clinical-procedure-biopsy = Biopsy
clinical-procedure-excision = Excision
clinical-procedure-resection = Resection
clinical-procedure-other = Other
clinical-grade = Grade
clinical-grade-system = Grading system
clinical-grade-value = Grade value
clinical-ptnm = pTNM
clinical-ptnm-t = T
clinical-ptnm-n = N
clinical-ptnm-m = M
clinical-ptnm-edition = AJCC edition
clinical-ptnm-prefix = Prefix (p/yp)
clinical-neoadjuvant = Neoadjuvant therapy
clinical-neoadjuvant-given = Given
clinical-neoadjuvant-regimen = Regimen
clinical-molecular = Molecular results
clinical-molecular-gene = Gene
clinical-molecular-result = Result
clinical-molecular-method = Method
clinical-specimen-month = Specimen month (YYYY-MM)
clinical-strip = Remove clinical data
clinical-export-with = Include clinical data in export

# --- Anatomical orientation ---
anat-panel = Anatomical orientation
anat-pos-x = +X direction
anat-pos-y = +Y direction
anat-pos-z = +Z direction (stack)
anat-anterior = Anterior
anat-posterior = Posterior
anat-superior = Superior
anat-inferior = Inferior
anat-medial = Medial
anat-lateral = Lateral
anat-proximal = Proximal
anat-distal = Distal
anat-left = Left
anat-right = Right
anat-unknown = Unknown

# --- Preanalytic (PreanalyticMeta) ---
preanalytic-panel = Preanalytic
preanalytic-block-id = Paraffin block ID
preanalytic-section-thickness-um = Section thickness (µm)
preanalytic-fixation-hours = Fixation time (hours)
preanalytic-fixative = Fixative
preanalytic-stained-month = Staining month (YYYY-MM)
preanalytic-scanner = Scanner
preanalytic-objective = Objective

# --- IHC panel (IhcPanel) ---
ihc-panel = IHC panel
ihc-required = The IHC panel is required for IHC-stained slides
ihc-antibody = Antibody
ihc-clone = Clone
ihc-chromogen = Chromogen
ihc-chromogen-dab = DAB (brown)
ihc-chromogen-aec = AEC (red)
ihc-chromogen-fast-red = Fast Red
ihc-chromogen-other = Other
ihc-pattern = Staining pattern
ihc-pattern-nuclear = Nuclear
ihc-pattern-membranous = Membranous
ihc-pattern-cytoplasmic = Cytoplasmic
ihc-scoring = Scoring scheme
ihc-scoring-percent-positive = Percent positive cells
ihc-scoring-h-score = H-score (0-300)
ihc-scoring-allred = Allred
ihc-scoring-tps = TPS (PD-L1 tumor cells)
ihc-scoring-cps = CPS (PD-L1 combined)
ihc-scoring-intensity03 = Intensity 0/1+/2+/3+
ihc-od-vectors = Stain OD vectors
ihc-od-default = Default (Ruifrok)
ihc-od-calibrated = Calibrated

# --- Label kinds (LabelKind::i18n_key) ---
label-panel = Labels
label-name = Name
label-color = Color
label-visible = Visible
label-reconstruct = Build mesh
label-kind = Kind
label-group = Group
label-kind-tumor = Tumor
label-kind-necrosis = Necrosis
label-kind-stroma = Stroma
label-kind-normal = Normal tissue
label-kind-vessel = Vessel
label-kind-vessel-blood = Blood vessel
label-kind-vessel-lymph = Lymph vessel
label-kind-vessel-unknown = Vessel (unknown)
label-kind-nerve = Nerve
label-kind-capsule = Capsule / serosa / fascia
label-kind-margin = Resection margin
label-kind-surface = Surface
label-kind-reference = Reference line
label-kind-other = Other

# --- Register mode ---
register-all = Register all
register-pair = Register this slice
register-qc-list = QC list
register-ncc = NCC
register-dice = Mask Dice
register-area-ratio = Area ratio
register-method = Method
register-flagged = Failed QC
register-flagged-count = { $n } slices failed QC
register-manual = Manual correction
register-landmarks = Landmarks
register-landmark-add = Add landmark
register-landmark-clear = Clear all landmarks
register-landmark-apply = Apply landmarks
register-landmark-hint = Click the same point on both slices (at least 3 pairs)
register-translate = Translate
register-rotate = Rotate
register-reset-transform = Reset transform
register-mark-damaged = Mark damaged
register-neighbor = Neighbor
register-compare-mode = Compare mode
register-compare-blend = Blend
register-compare-checkerboard = Checkerboard
register-compare-difference = Difference
register-blend-alpha = Blend opacity
register-checker-size = Checker size
register-progress = Registering { $done } of { $total }

# --- Segment mode ---
segment-tool-brush = Brush
segment-tool-eraser = Eraser
segment-tool-lasso = Lasso
segment-tool-polygon = Polygon
segment-brush-size = Brush size
segment-fill-holes = Fill holes
segment-copy-to-next = Copy to next slice
segment-copy-to-prev = Copy to previous slice
segment-overlay-opacity = Overlay opacity
segment-proposals = Proposals
segment-proposal-classic = Classic proposal
segment-proposal-ai = AI proposal
segment-proposal-interpolate = Interpolation proposal
segment-proposal-hint = Proposals never overwrite labels without Accept
segment-proposal-accept = Accept proposal
segment-proposal-reject = Reject proposal
segment-mark-reviewed = Mark as reviewed
segment-review-draft = Draft
segment-review-reviewed = Reviewed
segment-source-manual = Manual
segment-source-classic = Classic
segment-source-ai = AI
segment-source-interpolated = Interpolated
segment-calibrate-threshold = Calibrate IHC threshold
segment-threshold = Threshold
segment-positive-nuclei = Positive nuclei
segment-negative-nuclei = Negative nuclei
segment-finger-draw = Finger drawing
segment-pressure = Pressure

# --- Reconstruct mode ---
recon-run = Build reconstruction
recon-params = Parameters
recon-s-xy = XY voxel size (s_xy, µm)
recon-s-z = Z voxel size (s_z, µm)
recon-smoothing = Smoothing
recon-decimate = Decimate
recon-decimate-ratio = Decimation ratio
recon-opacity = Opacity
recon-visibility = Visibility
recon-clipping-plane = Clipping plane
recon-clipping-enable = Enable clipping plane
recon-clipping-axis = Clipping axis
recon-clipping-offset = Clipping position
recon-slices-in-3d = Slices in 3D
recon-show-slice = Show slice
recon-scalar-layer = Scalar layer
recon-scalar-none = No scalar layer
recon-legend = Legend
recon-colormap = Colormap
recon-colormap-viridis = Viridis
recon-colormap-magma = Magma
recon-colormap-coolwarm = Coolwarm
recon-colormap-grey = Grey
recon-range-min = Minimum
recon-range-max = Maximum
recon-range-auto = Automatic range (p2-p98)
recon-pick-value = Value: { $value } { $unit }
recon-pick-nearest-slice = Nearest slice: { $n }
recon-reset-camera = Reset camera
recon-mesh-stats = { $vertices } vertices, { $triangles } triangles

# --- Report mode ---
report-title = Report
report-export = Export report
report-export-csv = CSV
report-export-markdown = Markdown
report-export-mesh = Mesh (OBJ/PLY/STL)
report-export-nifti = NIfTI
report-section-1-context = 1. Sample context and orientation
report-section-2-components = 2. Tumor components
report-section-3-margin = 3. Margins
report-section-4-invasion = 4. Invasion
report-section-5-staging = 5. Staging size
report-section-6-attributes = 6. Attributes
report-section-7-provenance = 7. Provenance
report-section-8-params = 8. Parameters, versions, and warnings
report-measurements = Measurement table
report-warnings = Warnings
report-no-warnings = No warnings
report-generated-at = Generated { $time }
report-app-version = Application version { $version }
report-params-hash = Parameter hash { $hash }

measure-volume-cavalieri = Volume (Cavalieri)
measure-volume-mesh = Volume (mesh)
measure-volume-corrected = Shrinkage-corrected volume
measure-surface-area = Surface area
measure-bbox = Bounding box
measure-max-diameter = Maximum diameter
measure-pca-axes = Three PCA axes
measure-depth = Invasion depth
measure-breslow = Breslow thickness
measure-components = Component count
measure-foci = Focus count
measure-fraction = Fraction
measure-contact-area = Contact area
measure-patches = Patches
measure-uncertainty = Uncertainty
measure-margin-dmin = Minimum distance to margin
measure-margin-direction = Direction
measure-margin-nearest = Nearest point
measure-margin-status = Margin status
margin-status-involved = Involved
margin-status-close = Close
margin-status-clear = Clear
margin-status-unknown = Not assessable
measure-heterogeneity = Heterogeneity
measure-per-slice = Per slice
measure-per-region = Per region
measure-score-params = Score parameters
provenance-source-fraction = Slice fraction per source
provenance-unreviewed = Unreviewed slices
provenance-reviewed-only-volume = Volume from reviewed slices only

# --- Automatic warnings (8.12.8) ---
warn-margin-close = Tumor close to resection margin, direction { $dir }
warn-margin-z-unassessable = Distance to margin along Z cannot be assessed
warn-ihc-heterogeneity = High IHC heterogeneity; a single slice may not be representative
warn-mostly-interpolated = Most of the shape comes from interpolation
warn-has-draft = Figures include unreviewed annotations
warn-z-resolution-coarse = Z resolution is coarser than the margin threshold
warn-membrane-area-score = Score { $name } is computed per area, not per cell

# --- Errors ---
err-mpp-missing = mpp missing on slide { $name }; enter it manually
err-file-corrupt = File is corrupt or unreadable: { $name }
err-format-too-new = Project format is newer ({ $version }) than this application supports
err-singular = Singular transform; registration aborted
err-io = Read/write error: { $message }
err-unsupported-format = Unsupported file format: { $name }
err-ihc-panel-missing = IHC panel not filled for slide { $name }
err-no-labels = No labels to reconstruct yet
err-generic = An error occurred: { $message }
