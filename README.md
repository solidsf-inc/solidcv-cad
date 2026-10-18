# `solidcv-cad` — Pure-Rust Video-to-CAD Engine (`BRep` + Rational `NURBS` + `STEP AP242` + `IGES 5.3` + Binary `.glb` + `CAD -> CAE` Handoff)

`solidcv-cad` is a 100% pure-Rust, zero-FFI (`serde` + `serde_json` only), Apache-2.0 **Video-to-CAD** reconstruction and parametric solid-modeling engine developed by **solidSF** (**Andy Norris**, `andy@solidsf.com`).

Paired with [`solidcv`](https://github.com/solidsf-inc/solidcv) (2D/3D/4D Pixelean/Voxelean/Worldtube perception), [`solidcv-cae`](https://github.com/solidsf-inc/solidcv-cae) (Video-to-CAE full-field DIC/modal/thermal/CFD/FEA analysis), and [`ssftalk`](https://github.com/solidsf-inc/ssftalk) (Hamiltonian-Fourier voice & full-duplex rooms), `solidcv-cad` turns multi-frame video sequences directly into **watertight 2-manifold `BRep` solids**, **bicubic rational `NURBS` surface patches**, **ISO-10303-21 `STEP AP242`**, **`IGES 5.3`**, **binary `glTF 2.0` (`.glb`)**, and **watertight tetrahedral `CAD -> CAE` analysis decks**.

---

## Core Capabilities

1. **Watertight 2-Manifold `BRep` Reconstruction (`SolidCvBrepSolid`):**
   - Resolves single-view silhouette back-face ambiguity via 48-element octahedral $O_h$ symmetry completion.
   - Welds shared voxel boundary vertices and pairs opposite halfedges so every reconstructed shell satisfies `open_boundary_edges == 0`, `is_closed_manifold == true`, and Euler-Poincare characteristic $\chi = V - E + F = 2$.
2. **Bicubic Rational Tensor-Product `NURBS` Surface Fitting (`NurbsSurfacePatch`):**
   - Fits degree $(p=3, q=3)$ rational B-spline surface patches over clamped uniform knot vectors with exact recursive Cox-de Boor point and surface-normal evaluation.
3. **Principal Curvature Quadric Recognition (`PrincipalCurvatureQuadricFit`):**
   - Evaluates the first and second fundamental forms $(\kappa_1, \kappa_2, K = \kappa_1 \kappa_2, H = (\kappa_1 + \kappa_2)/2)$ to classify `PLANE`, `CYLINDER_OR_CONE`, `SPHERE`, `TORUS_SADDLE`, and `RATIONAL_NURBS`.
4. **Full Standards-Compliant Exports (`STEP AP242`, `IGES 5.3`, Binary `glTF 2.0 .glb`):**
   - `SolidCvBrepSolid::to_step_ap242()`: ISO-10303-21 `MANIFOLD_SOLID_BREP`, `CLOSED_SHELL`, `ADVANCED_FACE`, and `B_SPLINE_SURFACE_WITH_KNOTS`.
   - `export_brep_to_iges_5_3()`: Fixed 80-column IGES 5.3 (`S`, `G`, `D`, `P`, `T` sections) with `Entity 128` (Rational B-Spline Surface) and `Entity 186` (Manifold Solid B-Rep Object).
   - `SolidCvBrepSolid::to_glb_bytes()`: Binary `glTF 2.0` (`.glb`) with magic `0x46546C67` (`"glTF"`), `JSON` chunk `0x4E4F534A`, `BIN\0` chunk `0x004E4942`, and 4D worldtube trajectory keyframes.
5. **Direct `CAD -> CAE` Tetrahedral Mesh & Kinematic Handoff (`SolidCvCadToCaeHandoff`):**
   - `export_cad_to_cae_handoff()` generates a watertight 4-node linear tetrahedral mesh (`C3D4` / `VTK_TETRA`), kinematic velocity boundary conditions, and principal curvature stress-concentration priors for `solidcv-cae`.
6. **Cloud Binary WebSocket API (`wss://api.solidsf.com/v1/solidcv-cad/ws`) & Mechanistic Interpretability (`mechinterp`):**
   - Metered in **`solidSF Credits`** (`1 USD = 1,000,000 micro-USD`, `$10.00` welcome grant at [`https://solidsf.com/compute-credits`](https://solidsf.com/compute-credits)) with `GlassBoxCadSignature64D`, `SaeCadCircuitMatcher` (`b1.xor.popc`), and `with_counterfactual_ablation`.
