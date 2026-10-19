# solidSF Mechanical CAD & Multi-Agent Assembly Standard

**Document:** `SSF-CAD-ASSEMBLY`  
**Title:** solidSF Mechanical CAD & Multi-Agent Assembly Standard  
**Status:** Active (Revision 1.0)  
**Scope:** All 3D mechanical parts, multi-body hardware assemblies, parametric feature trees, kinematic solvers, and CAD/CAE pipelines across `solidasm`, `solidcv-cad`, and `solidsf` (`crates/solidasm/**`, `company/sites/jesse/solidasm_api.mjs`, `company/sites/jesse/solidcv_video_cad.mjs`).

---

## 1. Purpose

Every `solidSF` 3D part and multi-body assembly must be a genuine, mechanically sound, manufacturing-ready engineering artifact—never a decorative static mesh or collection of unfeatured primitive boxes. This standard establishes seven mandatory gates governing multi-agent, multi-pass decomposition, **ASME Y14.5-2018** geometric dimensioning and tolerancing (GD&T), **ISO 286-2** shaft/hole fits, live RPC part finalization, full editable parametric `FeatureTree` DAG retention, and true-perspective 3D visual verification.

---

## 2. Multi-Agent, Multi-Pass Pod Architecture (`Pass 1 -> Pass 2`)

Every hardware assembly project executes through a 5-clone Jesse engineering pod (`5` specialized Jesse roles per project) across a multi-pass refinement loop (`Pass 1` baseline topology and kinematic layout $\to$ `Pass 2` high-feature ASME Y14.5 mechanical refinement and live RPC finalization):

| Pod Clone Role | Engine Opcodes | Mandatory Engineering Responsibilities |
| :--- | :--- | :--- |
| **Clone +0: Vision, Blueprint & ASME Y14.5 Datum Intake** | `0x01, 0x02` | Extracts master geometry, wheelbase/envelope dimensions, and establishes primary (`Datum_A`), secondary (`Datum_B`), and tertiary (`Datum_C`) datum reference frames per **ASME Y14.5-2018** and **ISO 286-2** fits before solid synthesis begins. |
| **Clone +1: High-Genus BRep/NURBS & Steinmetz Coping** | `0x05, 0x06` | Synthesizes watertight 2-manifold BRep/NURBS solids (`V - E + F = 2(1 - g)`) with analytical Steinmetz bicubic NURBS fishmouth coping (`max_gap_mm <= 0.08`) at all tube/strut joints and G2 curvature-continuous lofted fillets. |
| **Clone +2: $SE(3)$ Kinematic Loop Closure & Coaxial Mating** | `0x07` | Solves exact $SE(3)$ assembly mates (`Revolute`, `Cylindrical`, `Planar`, `Spherical`, `GearRack`, `HelicalScrew`, `RigidWeld`) with `residual_norm <= 1e-9` and verified Grübler-Kutzbach mobility DOF. |
| **Clone +3: ASME Y14.5 Tolerance Stackup & Mass/Inertia BOM** | `0x08, 0x09` | Computes worst-case and RSS $3\sigma$ tolerance chains (`Cpk >= 1.33`, `PASS` verdict) with explicit ISO 286 shaft/bore fits (`H7/g6`, `H7/p6`, `JS6/k5`) and full physical `AssemblyBom` (material density, mass in kg, center of mass, principal inertia tensor, and exploded-view vectors). |
| **Clone +4: Live RPC Part Finalization & Multi-Format Export** | `0x0A` | Finalizes every part and assembly via live `SolidAsmLoopbackWsServer` RPC calls and emits `ISO-10303-21 STEP AP242`, binary `glTF 2.0 .glb`, `IGES 5.3 .igs`, `*_editable_feature_tree.json`, and `SolidAsmToCaeHandoff`. |

---

## 3. The Seven Mandatory Gates

### Gate 1: ASME Y14.5-2018 Datum Reference Frame & ISO 286 Fit Alignment
- Every assembly and part must declare explicit primary, secondary, and tertiary datums (`Datum_A`, `Datum_B`, `Datum_C`) and reference all child sketches, bores, and mating surfaces to that datum hierarchy.
- Cylindrical bearing seats, press fits, and sliding joints must specify standard **ISO 286-2** tolerance classes (e.g., `6806_2RS_30x42x7_H7_k5`, `H7/g6` precision running fit, `H7/p6` locational interference fit) and pass both worst-case and RSS $3\sigma$ stackup (`Cpk >= 1.33`).

### Gate 2: Live RPC Part Finalization (`feature_tree.finalize_part_brep`)
- Static geometry generation that bypasses the `SolidAsmLoopbackWsServer` RPC engine is prohibited.
- Every part in the assembly must be finalized by invoking `feature_tree.finalize_part_brep` (opcode `0x05`, `25 uUSD` per part), followed by assembly-level `assemble_step_ap242` (`0x0A`), `validate_tolerance_bom` (`0x08`), and `solidcv_cae_handoff` (`0x0A`), recording a deterministic `*_rpc_receipt.json` with per-call latency and micro-USD credit debits.

### Gate 3: Full Editable Parametric `FeatureTree` DAG & Live Post-Build Mutation
- Every part must retain its complete, ordered, non-baked `PartEditableFeatureTree` DAG (`20–200+` parametric `EditableFeatureNode` operations per complex part) with explicit `parent_dependencies` and named scalar parameters (`BTreeMap<String, f64>`).
- Supported parametric feature operations include: `DatumPlane`, `CoordinateSystem`, `MasterProfileSketch`, `GuideCurve3DSketch`, `MultiSectionLoft`, `SolidExtrudeBoss`, `RevolveBoss`, `HelicalSweepCut`, `SteinmetzTubeCopeCut`, `CircularFeaturePattern`, `LinearFeaturePattern`, `VariableRadiusFillet`, `ConstantEdgeChamfer`, `ShellThinWall`, `BooleanUnion`, `BooleanSubtract`, `CylindricalBoreCut`, `CounterboreHoleWizard`, `ThreadCosmeticHelix`, `KinematicMateConnector`, and `MassPropertySensor`.
- Modifying any upstream node parameter via RPC (`feature_tree.mutate_param`) must propagate `dirty = true` strictly along downstream dependent DAG edges, re-evaluate the affected BRep solid (`Rev-A -> Rev-B`), update mass/vertex/triangle telemetry, and preserve Euler characteristic $V - E + F = 2$.

### Gate 4: Zero Primitive Box Placeholders & Minimum Feature/Vertex Complexity
- **Primitive 8-vertex or 14-vertex rectangular boxes and un-coped overlapping cylinders are strictly prohibited.**
- Every structural tube, bracket, housing, gear, wheel, linkage, or aerodynamic fairing must include its functional mechanical features: precision pilot bores, bolt-circle counterbores, involute gear/sprocket teeth, lightening pockets, chamfers, variable-radius fillets, and wall-thickness shells (`>= 48` vertices per part in multi-project swarms; `20–200+` features per part in hero assemblies).

### Gate 5: Coaxial Bore-to-Pin Alignment & Kinematic Loop Closure
- Every `Revolute`, `Cylindrical`, `Spherical`, and `HelicalScrew` joint must pair a physical bore/bearing race on Part A with a coaxial shaft/pin/journal on Part B at identical world-frame $SE(3)$ coordinates in the assembled state (`residual_norm <= 1e-9`), with zero solid interference across the joint's range of motion.

### Gate 6: True-Perspective 3D PBR Visual & Video Verification
- Every delivered assembly or multi-project showcase must be rendered through a true 3D perspective camera (`1920x1080` WebGL2/Three.js PBR or native rasterizer with normal/Fresnel shading, studio grid, and engineering HUD callouts) across four mandatory views:
  1. Orthographic / telephoto side engineering profile
  2. $3/4$ perspective hero orbit
  3. Close-up drivetrain / joint / mechanism detail
  4. Exploded BOM assembly view

### Gate 7: Complete Multi-Format Artifact Bundle & Programmatic Gate (`assert_ssf_cad_assembly_standard`)
- Every assembly output directory must contain:
  - `<id>.step` (`ISO-10303-21;` STEP AP242 with `NEXT_ASSEMBLY_USAGE_OCCURRENCE`, `MANIFOLD_SOLID_BREP`, and `B_SPLINE_SURFACE_WITH_KNOTS`)
  - `<id>.glb` (binary `glTF 2.0` starting with `0x46546C67` and named part nodes)
  - `<id>.igs` (`IGES 5.3` with `Type 128` NURBS surfaces)
  - `<id>_editable_feature_tree.json` (`solidasm.editable_feature_tree.v2`)
  - `<id>_bom.json` & `<id>_tolerance_stackup.json`
  - `<id>_rpc_receipt.json`
- Every assembly must pass `solidasm::roadbike_dag::assert_ssf_cad_assembly_standard` with zero errors and zero compiler warnings.
