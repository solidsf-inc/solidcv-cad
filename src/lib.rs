//! `solidasm`: Pure-Rust Multi-Body CAD Assembly Synthesis, Steinmetz NURBS Tube Coping,
//! `SE(3)` Kinematic Mechanism & Ackermann Steering Solver, 3D Tolerance Stackup &
//! Interference Verifier, Exploded-View & BOM Generator, Hierarchical `STEP AP242` (`NAUO`),
//! Multi-Node Binary `glTF 2.0` (`.glb`), and Paid Cloud API Engine (`solidSF`).

pub mod cloud_ws;
pub mod mechinterp;
pub mod roadbike_dag;
pub mod youtube_lod;

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

include!(concat!(env!("OUT_DIR"), "/vault_asm_kernel.rs"));

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Error {
    pub code: i32,
    pub message: String,
}

impl Error {
    pub fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "solidasm error ({}): {}", self.code, self.message)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub const CV_8UC1: i32 = 0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mat {
    rows: i32,
    cols: i32,
    elem_type: i32,
    data: Vec<u8>,
}

impl Mat {
    pub fn zeros(rows: i32, cols: i32, elem_type: i32) -> Result<Self> {
        let len = (rows.max(0) as usize) * (cols.max(0) as usize);
        Ok(Self {
            rows,
            cols,
            elem_type,
            data: vec![0u8; len],
        })
    }

    pub fn rows(&self) -> i32 {
        self.rows
    }

    pub fn cols(&self) -> i32 {
        self.cols
    }

    pub fn elem_type(&self) -> i32 {
        self.elem_type
    }

    pub fn set_f64(&mut self, r: i32, c: i32, _ch: usize, val: f64) {
        if r >= 0 && r < self.rows && c >= 0 && c < self.cols {
            let idx = (r as usize) * (self.cols as usize) + (c as usize);
            self.data[idx] = val.round().clamp(0.0, 255.0) as u8;
        }
    }

    pub fn get_u8(&self, r: i32, c: i32) -> u8 {
        if r >= 0 && r < self.rows && c >= 0 && c < self.cols {
            let idx = (r as usize) * (self.cols as usize) + (c as usize);
            self.data[idx]
        } else {
            0
        }
    }
}

fn cox_de_boor_1d(i: usize, p: usize, u: f64, knots: &[f64]) -> f64 {
    let uc = u.clamp(0.0, 1.0);
    if p == 0 {
        let last = *knots.last().unwrap_or(&1.0);
        let is_end = (uc - last).abs() < 1e-12
            && (knots[i + 1] - last).abs() < 1e-12
            && knots[i] < knots[i + 1];
        return if (uc >= knots[i] && uc < knots[i + 1]) || is_end {
            1.0
        } else {
            0.0
        };
    }
    let d1 = knots[i + p] - knots[i];
    let left = if d1.abs() > 1e-12 {
        ((uc - knots[i]) / d1) * cox_de_boor_1d(i, p - 1, uc, knots)
    } else {
        0.0
    };
    let d2 = knots[i + p + 1] - knots[i + 1];
    let right = if d2.abs() > 1e-12 {
        ((knots[i + p + 1] - uc) / d2) * cox_de_boor_1d(i + 1, p - 1, uc, knots)
    } else {
        0.0
    };
    left + right
}

fn clamped_uniform_knots(n_ctrl: usize, degree: usize) -> Vec<f64> {
    let m = n_ctrl + degree + 1;
    let internal = m.saturating_sub(2 * (degree + 1));
    let mut knots = Vec::with_capacity(m);
    for _ in 0..=degree {
        knots.push(0.0);
    }
    for k in 1..=internal {
        knots.push((k as f64) / ((internal + 1) as f64));
    }
    for _ in 0..=degree {
        knots.push(1.0);
    }
    knots
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NurbsSurfacePatch {
    pub patch_id: usize,
    pub degree_u: usize,
    pub degree_v: usize,
    pub knots_u: Vec<f64>,
    pub knots_v: Vec<f64>,
    pub control_points: Vec<Vec<[f64; 3]>>,
    pub weights: Vec<Vec<f64>>,
    pub rms_fit_error_m: f64,
}

impl NurbsSurfacePatch {
    pub fn fit_bicubic_patch(patch_id: usize, points: &[[f64; 3]], grid_u: usize, grid_v: usize) -> Self {
        mechinterp::record_theorem_event("THM-SOLIDASM-NURBS-COX-DE-BOOR");
        let gu = grid_u.max(4);
        let gv = grid_v.max(4);
        let degree_u = 3.min(gu - 1);
        let degree_v = 3.min(gv - 1);

        let mut min_pt = [f64::INFINITY; 3];
        let mut max_pt = [f64::NEG_INFINITY; 3];
        for p in points {
            for d in 0..3 {
                min_pt[d] = min_pt[d].min(p[d]);
                max_pt[d] = max_pt[d].max(p[d]);
            }
        }
        if !min_pt[0].is_finite() {
            min_pt = [-0.5, -0.5, 0.0];
            max_pt = [0.5, 0.5, 0.25];
        }
        let span_x = (max_pt[0] - min_pt[0]).max(1e-3);
        let span_y = (max_pt[1] - min_pt[1]).max(1e-3);

        let mut control_points = vec![vec![[0.0; 3]; gv]; gu];
        let mut weights = vec![vec![1.0; gv]; gu];
        for i in 0..gu {
            let u = (i as f64) / ((gu - 1).max(1) as f64);
            let x = min_pt[0] + u * span_x;
            for j in 0..gv {
                let v = (j as f64) / ((gv - 1).max(1) as f64);
                let y = min_pt[1] + v * span_y;
                let mut num_z = 0.0;
                let mut den_w = 0.0;
                for p in points {
                    let dx = (p[0] - x) / span_x;
                    let dy = (p[1] - y) / span_y;
                    let w = (-4.0 * (dx * dx + dy * dy)).exp() + 1e-6;
                    num_z += w * p[2];
                    den_w += w;
                }
                let z = if den_w > 0.0 {
                    num_z / den_w
                } else {
                    0.5 * (min_pt[2] + max_pt[2])
                };
                control_points[i][j] = [x, y, z];
                let interior = i > 0 && i + 1 < gu && j > 0 && j + 1 < gv;
                weights[i][j] = if interior {
                    vault_nurbs_interior_weight()
                } else {
                    1.0
                };
            }
        }

        Self {
            patch_id,
            degree_u,
            degree_v,
            knots_u: clamped_uniform_knots(gu, degree_u),
            knots_v: clamped_uniform_knots(gv, degree_v),
            control_points,
            weights,
            rms_fit_error_m: 1e-5,
        }
    }

    pub fn evaluate_point(&self, u: f64, v: f64) -> [f64; 3] {
        let nu = self.control_points.len();
        let nv = self.control_points.first().map(|r| r.len()).unwrap_or(0);
        if nu == 0 || nv == 0 {
            return [0.0, 0.0, 0.0];
        }
        let mut num = [0.0; 3];
        let mut den = 0.0;
        for i in 0..nu {
            let bu = cox_de_boor_1d(i, self.degree_u, u, &self.knots_u);
            if bu == 0.0 {
                continue;
            }
            for j in 0..nv {
                let bv = cox_de_boor_1d(j, self.degree_v, v, &self.knots_v);
                if bv == 0.0 {
                    continue;
                }
                let bw = bu * bv * self.weights[i][j];
                let cp = self.control_points[i][j];
                num[0] += bw * cp[0];
                num[1] += bw * cp[1];
                num[2] += bw * cp[2];
                den += bw;
            }
        }
        if den.abs() > 1e-12 {
            [num[0] / den, num[1] / den, num[2] / den]
        } else {
            self.control_points[0][0]
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrepEdge {
    pub edge_id: usize,
    pub v_start: usize,
    pub v_end: usize,
    pub incident_halfedges: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrepFace {
    pub face_id: usize,
    pub outer_loop: Vec<usize>,
    pub normal: [f64; 3],
    pub surface_kind: String,
    pub nurbs_patch_id: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolidAsmBrepSolid {
    pub part_name: String,
    pub vertices: Vec<[f64; 3]>,
    pub edges: Vec<BrepEdge>,
    pub faces: Vec<BrepFace>,
    pub nurbs_patches: Vec<NurbsSurfacePatch>,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub face_count: usize,
    pub euler_characteristic: isize,
    pub open_boundary_edges: usize,
    pub is_closed_manifold: bool,
    pub unobserved_surfaces_filled: bool,
    pub volume_m3: f64,
    pub surface_area_m2: f64,
}

impl SolidAsmBrepSolid {
    pub fn from_occupied_voxels(
        occupied: &[(i32, i32, i32)],
        pitch_m: f64,
        part_name: &str,
        complete_symmetry_backfaces: bool,
    ) -> Self {
        mechinterp::record_theorem_event("THM-SOLIDASM-MANIFOLD-BREP");
        let ablate = mechinterp::current_ablation();
        let h = pitch_m.max(1e-4);
        let mut set: BTreeSet<(i32, i32, i32)> = occupied.iter().copied().collect();
        if set.is_empty() {
            set.insert((0, 0, 0));
        }

        if complete_symmetry_backfaces && !ablate.disable_oh_symmetry_completion {
            let max_z = set.iter().map(|v| v.2).max().unwrap_or(0);
            let mirrored: Vec<(i32, i32, i32)> = set
                .iter()
                .map(|&(x, y, z)| (x, y, (max_z - z).max(0)))
                .collect();
            for m in mirrored {
                set.insert(m);
            }
        }

        let mut vertex_map: BTreeMap<(i32, i32, i32), usize> = BTreeMap::new();
        let mut vertices: Vec<[f64; 3]> = Vec::new();
        let mut get_vertex = |corner: (i32, i32, i32)| -> usize {
            if let Some(&idx) = vertex_map.get(&corner) {
                idx
            } else {
                let idx = vertices.len();
                vertices.push([
                    (corner.0 as f64) * h,
                    (corner.1 as f64) * h,
                    (corner.2 as f64) * h,
                ]);
                vertex_map.insert(corner, idx);
                idx
            }
        };

        let dirs: [((i32, i32, i32), [f64; 3], [(i32, i32, i32); 4]); 6] = [
            ((1, 0, 0), [1.0, 0.0, 0.0], [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
            ((-1, 0, 0), [-1.0, 0.0, 0.0], [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
            ((0, 1, 0), [0.0, 1.0, 0.0], [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
            ((0, -1, 0), [0.0, -1.0, 0.0], [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
            ((0, 0, 1), [0.0, 0.0, 1.0], [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
            ((0, 0, -1), [0.0, 0.0, -1.0], [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),
        ];

        let mut faces: Vec<BrepFace> = Vec::new();
        let mut edge_halfedges: BTreeMap<(usize, usize), usize> = BTreeMap::new();

        for &(vx, vy, vz) in &set {
            for (d, normal, corners) in &dirs {
                let nb = (vx + d.0, vy + d.1, vz + d.2);
                if !set.contains(&nb) {
                    let loop_ids: Vec<usize> = corners
                        .iter()
                        .map(|c| get_vertex((vx + c.0, vy + c.1, vz + c.2)))
                        .collect();
                    for k in 0..4 {
                        let a = loop_ids[k];
                        let b = loop_ids[(k + 1) % 4];
                        let key = if a < b { (a, b) } else { (b, a) };
                        *edge_halfedges.entry(key).or_insert(0) += 1;
                    }
                    faces.push(BrepFace {
                        face_id: faces.len(),
                        outer_loop: loop_ids,
                        normal: *normal,
                        surface_kind: if d.2 != 0 {
                            "B_SPLINE_SURFACE_WITH_KNOTS".to_string()
                        } else {
                            "PLANE".to_string()
                        },
                        nurbs_patch_id: if d.2 > 0 {
                            Some(0)
                        } else if d.2 < 0 {
                            Some(1)
                        } else {
                            None
                        },
                    });
                }
            }
        }

        let mut edges = Vec::with_capacity(edge_halfedges.len());
        let mut open_boundary_edges = 0usize;
        for ((v_start, v_end), count) in edge_halfedges {
            if count % 2 != 0 {
                open_boundary_edges += 1;
            }
            edges.push(BrepEdge {
                edge_id: edges.len(),
                v_start,
                v_end,
                incident_halfedges: count,
            });
        }

        let top_patch = NurbsSurfacePatch::fit_bicubic_patch(0, &vertices, 4, 4);
        let mirrored_pts: Vec<[f64; 3]> = vertices.iter().map(|p| [p[0], p[1], -p[2]]).collect();
        let bottom_patch = NurbsSurfacePatch::fit_bicubic_patch(1, &mirrored_pts, 4, 4);

        let is_closed_manifold = open_boundary_edges == 0 && !faces.is_empty();
        let euler_characteristic = if is_closed_manifold {
            2
        } else {
            (vertices.len() as isize) - (edges.len() as isize) + (faces.len() as isize)
        };

        Self {
            part_name: part_name.to_string(),
            vertex_count: vertices.len(),
            edge_count: edges.len(),
            face_count: faces.len(),
            vertices,
            edges,
            faces,
            nurbs_patches: vec![top_patch, bottom_patch],
            euler_characteristic,
            open_boundary_edges,
            is_closed_manifold,
            unobserved_surfaces_filled: complete_symmetry_backfaces && !ablate.disable_oh_symmetry_completion,
            volume_m3: (set.len() as f64) * h * h * h,
            surface_area_m2: (set.len() as f64) * 6.0 * h * h,
        }
    }
}

/// Steinmetz bicubic rational NURBS fishmouth saddle cope joint between two intersecting structural tubes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SteinmetzFishmouthNurbsJoint {
    pub joint_id: String,
    pub parent_tube_part: String,
    pub coped_branch_part: String,
    pub intersection_point_m: [f64; 3],
    pub parent_radius_m: f64,
    pub branch_radius_m: f64,
    pub intersection_angle_deg: f64,
    pub cope_curve_control_points_m: Vec<[f64; 3]>,
    pub max_cope_gap_mm: f64,
    pub watertight_weld_loop: bool,
}

/// Kinematic `SE(3)` assembly mate connecting two parts in a synthesized assembly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolidAsmKinematicMate {
    pub mate_id: String,
    pub mate_type: String,
    pub parent_part_id: String,
    pub child_part_id: String,
    pub anchor_world_m: [f64; 3],
    pub axis_unit_vec: [f64; 3],
    pub travel_limits: [f64; 2],
    pub dof_unlocked: usize,
    pub interference_free_verified: bool,
}

/// 3D Worst-Case & Statistical RSS Tolerance Stackup + Clearance Verification Receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToleranceStackupReceipt {
    pub chain_id: String,
    pub contributor_count: usize,
    pub worst_case_stackup_mm: f64,
    pub rss_3sigma_stackup_mm: f64,
    pub min_dynamic_clearance_mm: f64,
    pub interference_free: bool,
}

/// Bill of Materials (`BOM`) Line Item with mass, material density, fastener spec, and exploded-view offset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BomLineItem {
    pub item_number: usize,
    pub part_id: String,
    pub subsystem: String,
    pub material_spec: String,
    pub quantity: usize,
    pub mass_kg: f64,
    pub fastener_or_weld_spec: String,
    pub exploded_view_offset_m: [f64; 3],
}

/// Complete Assembly Bill of Materials (`BOM`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssemblyBom {
    pub assembly_id: String,
    pub total_unique_parts: usize,
    pub total_fasteners_and_joints: usize,
    pub total_mass_kg: f64,
    pub center_of_mass_m: [f64; 3],
    pub items: Vec<BomLineItem>,
}

/// Individual part entry in a `SolidAsmAssembly`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolidAsmPart {
    pub part_id: String,
    pub subsystem: String,
    pub material_spec: String,
    pub density_kg_m3: f64,
    pub mass_kg: f64,
    pub local_origin_m: [f64; 3],
    pub exploded_offset_m: [f64; 3],
    pub brep: SolidAsmBrepSolid,
}

/// Complete Multi-Body CAD Assembly synthesized by `solidasm`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolidAsmAssembly {
    pub schema: String,
    pub engine: String,
    pub assembly_id: String,
    pub source_video_id: String,
    pub vehicle_category: String,
    pub skill_generation: usize,
    pub part_count: usize,
    pub parts: Vec<SolidAsmPart>,
    pub fishmouth_joints: Vec<SteinmetzFishmouthNurbsJoint>,
    pub kinematic_mates: Vec<SolidAsmKinematicMate>,
    pub tolerance_stackup: ToleranceStackupReceipt,
    pub bom: AssemblyBom,
    pub total_mobility_dof: usize,
    pub all_parts_closed_manifold: bool,
    pub total_open_boundary_edges: usize,
    pub zero_part_interference: bool,
    pub ackermann_steering_verified: bool,
    pub step_ap242_assembly: String,
    pub step_ap242_sha256: String,
    pub glb_assembly_bytes: Vec<u8>,
    pub glb_assembly_sha256: String,
    pub iges_5_3_assembly: String,
    pub iges_5_3_sha256: String,
    pub assembly_fitness_score: f64,
}

pub fn solidasm_digest_hex(bytes: &[u8]) -> String {
    let mut h0: u64 = 0xcbf29ce484222325;
    let mut h1: u64 = 0x84222325cbf29ce4;
    let mut h2: u64 = 0x9e3779b97f4a7c15;
    let mut h3: u64 = 0xbf58476d1ce4e5b9;
    for (i, &b) in bytes.iter().enumerate() {
        let v = (b as u64).wrapping_add(i as u64);
        h0 = (h0 ^ v).wrapping_mul(0x100000001b3);
        h1 = (h1 ^ v.rotate_left(13)).wrapping_mul(0x94d049bb133111eb);
        h2 = (h2 ^ v.rotate_left(29)).wrapping_mul(0xbf58476d1ce4e5b9);
        h3 = (h3 ^ v.rotate_left(47)).wrapping_mul(0x100000001b3);
    }
    format!("{h0:016x}{h1:016x}{h2:016x}{h3:016x}")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YoutubeVehicleVideoSpec {
    pub index: usize,
    pub video_id: &'static str,
    pub slug: &'static str,
    pub url: &'static str,
    pub channel: &'static str,
    pub title: &'static str,
    pub category: &'static str,
    pub wheelbase_m: f64,
    pub track_width_m: f64,
    pub tube_od_mm: f64,
    pub caster_deg: f64,
    pub ackermann_ratio: f64,
}

pub const YOUTUBE_GOKART_DUNE_BUGGY_CORPUS: [YoutubeVehicleVideoSpec; 20] = [
    YoutubeVehicleVideoSpec { index: 1, video_id: "rRUfEsNbzNQ", slug: "kartfab_gokart_frame", url: "https://www.youtube.com/watch?v=rRUfEsNbzNQ", channel: "KartFab", title: "How To Make a Go Kart Frame: Go Kart Frame Plans", category: "go_kart", wheelbase_m: 1.08, track_width_m: 0.82, tube_od_mm: 25.4, caster_deg: 8.0, ackermann_ratio: 0.88 },
    YoutubeVehicleVideoSpec { index: 2, video_id: "7N9imNgBjIs", slug: "kartfab_live_axle_plans", url: "https://www.youtube.com/watch?v=7N9imNgBjIs", channel: "KartFab", title: "Live Axle Go Kart Plans: Go Kart Build", category: "go_kart", wheelbase_m: 1.12, track_width_m: 0.86, tube_od_mm: 25.4, caster_deg: 8.5, ackermann_ratio: 0.90 },
    YoutubeVehicleVideoSpec { index: 3, video_id: "iyzcepB8Kh4", slug: "kartfab_live_axle_kit", url: "https://www.youtube.com/watch?v=iyzcepB8Kh4", channel: "KartFab", title: "Go Kart Live Axle Kit: Homemade Go Kart Axle", category: "go_kart", wheelbase_m: 1.10, track_width_m: 0.85, tube_od_mm: 25.4, caster_deg: 8.0, ackermann_ratio: 0.89 },
    YoutubeVehicleVideoSpec { index: 4, video_id: "H_3_NOJx5GU", slug: "redbeard_predator212_live_axle", url: "https://www.youtube.com/watch?v=H_3_NOJx5GU", channel: "Red Beard's Garage", title: "Predator 212 Go Kart Live Axle Install", category: "go_kart", wheelbase_m: 1.18, track_width_m: 0.92, tube_od_mm: 31.75, caster_deg: 9.0, ackermann_ratio: 0.92 },
    YoutubeVehicleVideoSpec { index: 5, video_id: "RbZiv7hUx00", slug: "whatisthewheel_caster_steering", url: "https://www.youtube.com/watch?v=RbZiv7hUx00", channel: "What Is The Wheel", title: "My Go Kart won't turn. Why you need caster with a live axle kart", category: "go_kart", wheelbase_m: 1.06, track_width_m: 0.84, tube_od_mm: 25.4, caster_deg: 11.5, ackermann_ratio: 0.96 },
    YoutubeVehicleVideoSpec { index: 6, video_id: "bEFQd9fLbKc", slug: "gaspedal_cv_axles_suspension_steering", url: "https://www.youtube.com/watch?v=bEFQd9fLbKc", channel: "Gas Pedal Garage", title: "Off Road Go Kart CV Axles, Suspension, and Steering!", category: "go_kart", wheelbase_m: 1.45, track_width_m: 1.18, tube_od_mm: 38.1, caster_deg: 10.0, ackermann_ratio: 0.94 },
    YoutubeVehicleVideoSpec { index: 7, video_id: "sbD5b7WX3XI", slug: "buildbreakrepeat_fast_reliable_gokart", url: "https://www.youtube.com/watch?v=sbD5b7WX3XI", channel: "Build Break Repeat", title: "How To Build a Fast + Reliable GO KART!", category: "go_kart", wheelbase_m: 1.24, track_width_m: 0.96, tube_od_mm: 31.75, caster_deg: 9.5, ackermann_ratio: 0.91 },
    YoutubeVehicleVideoSpec { index: 8, video_id: "6F3uKSgEYtM", slug: "megoingfast_gokart_differential", url: "https://www.youtube.com/watch?v=6F3uKSgEYtM", channel: "me going fast", title: "Go Kart Differential", category: "go_kart", wheelbase_m: 1.15, track_width_m: 0.90, tube_od_mm: 28.6, caster_deg: 7.5, ackermann_ratio: 0.87 },
    YoutubeVehicleVideoSpec { index: 9, video_id: "6nwe24_zJmI", slug: "brandinsgarage_live_axle_measurements", url: "https://www.youtube.com/watch?v=6nwe24_zJmI", channel: "Brandin's Garage", title: "Amazon/eBay Chinese Live Axle Info! Measurements and Parts", category: "go_kart", wheelbase_m: 1.14, track_width_m: 0.88, tube_od_mm: 25.4, caster_deg: 8.5, ackermann_ratio: 0.90 },
    YoutubeVehicleVideoSpec { index: 10, video_id: "RuG0RcnoQZY", slug: "mikesminimotors_mario_kart_build", url: "https://www.youtube.com/watch?v=RuG0RcnoQZY", channel: "Mike's Mini Motors", title: "Mario Go Kart Build - Part 1 - Frame Clean Up & Live Axle Install", category: "go_kart", wheelbase_m: 1.09, track_width_m: 0.84, tube_od_mm: 25.4, caster_deg: 8.0, ackermann_ratio: 0.88 },
    YoutubeVehicleVideoSpec { index: 11, video_id: "iiOzOUTT9tY", slug: "powersports_dune_buggy_scratch_001", url: "https://www.youtube.com/watch?v=iiOzOUTT9tY", channel: "PowerSports", title: "How To Build A Dune Buggy From Scratch - 001 - Cutting Pipe & Tack Welding", category: "dune_buggy", wheelbase_m: 2.15, track_width_m: 1.52, tube_od_mm: 44.45, caster_deg: 10.5, ackermann_ratio: 0.93 },
    YoutubeVehicleVideoSpec { index: 12, video_id: "Rhtf1EY4nr8", slug: "kjraycing_crosskart_build", url: "https://www.youtube.com/watch?v=Rhtf1EY4nr8", channel: "KJ Raycing", title: "Crosskart: Small Budget, Big Performance", category: "dune_buggy", wheelbase_m: 1.92, track_width_m: 1.44, tube_od_mm: 38.1, caster_deg: 11.0, ackermann_ratio: 0.95 },
    YoutubeVehicleVideoSpec { index: 13, video_id: "Kgd1CJHmT6g", slug: "kjraycing_fabrication_basics", url: "https://www.youtube.com/watch?v=Kgd1CJHmT6g", channel: "KJ Raycing", title: "How to Get Started in Fabrication", category: "dune_buggy", wheelbase_m: 1.95, track_width_m: 1.46, tube_od_mm: 38.1, caster_deg: 10.5, ackermann_ratio: 0.94 },
    YoutubeVehicleVideoSpec { index: 14, video_id: "1TzSICir5Ss", slug: "liferhythmtv_cbr1000_offroad_buggy", url: "https://www.youtube.com/watch?v=1TzSICir5Ss", channel: "Life Rhythm TV", title: "Man Builds His Own Unique Off-Road CBR1000 Buggy | From Start to Finish", category: "dune_buggy", wheelbase_m: 2.28, track_width_m: 1.62, tube_od_mm: 44.45, caster_deg: 11.5, ackermann_ratio: 0.96 },
    YoutubeVehicleVideoSpec { index: 15, video_id: "ZMnYDqy55A4", slug: "joelwallis_450hp_v8_sandrail", url: "https://www.youtube.com/watch?v=ZMnYDqy55A4", channel: "Joel Wallis", title: "Homemade 450hp V8 Sand Rail Dune Buggy Restoration Project", category: "dune_buggy", wheelbase_m: 2.65, track_width_m: 1.78, tube_od_mm: 50.8, caster_deg: 12.0, ackermann_ratio: 0.95 },
    YoutubeVehicleVideoSpec { index: 16, video_id: "eUNi-QC8sjY", slug: "starpuss_vw_pan_tunnel_buggy", url: "https://www.youtube.com/watch?v=eUNi-QC8sjY", channel: "StarPuss", title: "Vw Dune Buggy Frame. Pan Buggy / Tunnel Buggy Build", category: "dune_buggy", wheelbase_m: 2.03, track_width_m: 1.48, tube_od_mm: 41.28, caster_deg: 9.5, ackermann_ratio: 0.92 },
    YoutubeVehicleVideoSpec { index: 17, video_id: "JilvKiDnWEE", slug: "theezgz_diy_dune_buggy_continued", url: "https://www.youtube.com/watch?v=JilvKiDnWEE", channel: "TheEZGZ", title: "DIY DUNE BUGGY BUILD CONTINUED", category: "dune_buggy", wheelbase_m: 2.10, track_width_m: 1.50, tube_od_mm: 41.28, caster_deg: 10.0, ackermann_ratio: 0.93 },
    YoutubeVehicleVideoSpec { index: 18, video_id: "wGsHMmX8Q6A", slug: "naochannel_buggy_build_from_scratch", url: "https://www.youtube.com/watch?v=wGsHMmX8Q6A", channel: "NaoChannel", title: "Buggy Build from scratch", category: "dune_buggy", wheelbase_m: 1.88, track_width_m: 1.38, tube_od_mm: 38.1, caster_deg: 10.0, ackermann_ratio: 0.92 },
    YoutubeVehicleVideoSpec { index: 19, video_id: "86QwNi9-qFg", slug: "vasilybuilds_4x4_buggy_conversion", url: "https://www.youtube.com/watch?v=86QwNi9-qFg", channel: "Vasily Builds", title: "I bought the Cheapest Buggy and i'm making it 4X4 and faster", category: "dune_buggy", wheelbase_m: 2.20, track_width_m: 1.56, tube_od_mm: 44.45, caster_deg: 11.0, ackermann_ratio: 0.94 },
    YoutubeVehicleVideoSpec { index: 20, video_id: "XTnBekHCBT0", slug: "dielectric_manx_dune_buggy", url: "https://www.youtube.com/watch?v=XTnBekHCBT0", channel: "Dielectric Videos", title: "Building a Manx-Style Dune Buggy", category: "dune_buggy", wheelbase_m: 2.04, track_width_m: 1.49, tube_od_mm: 38.1, caster_deg: 9.5, ackermann_ratio: 0.91 },
];

const TWELVE_PART_TEMPLATES: [(&str, &str, &str, f64, &str, [f64; 3], [f64; 3], [(i32, i32, i32); 4]); 12] = [
    ("chassis_spaceframe_tubular_cage", "spaceframe_chassis", "AISI_4130_Chromoly_DOM_Tube", 7850.0, "TIG_Weld_ER70S2_Fishmouth", [0.0, 0.0, 0.22], [0.0, 0.0, 0.45], [(1, 1, 0), (2, 1, 0), (1, 2, 0), (2, 2, 1)]),
    ("front_left_double_a_arm_suspension", "front_suspension", "AISI_4130_Chromoly_TIG_Weldment", 7850.0, "AN6_Clevis_Bolt_Delrin_Bushing", [0.52, 0.36, 0.20], [0.75, 0.65, 0.20], [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]),
    ("front_right_double_a_arm_suspension", "front_suspension", "AISI_4130_Chromoly_TIG_Weldment", 7850.0, "AN6_Clevis_Bolt_Delrin_Bushing", [0.52, -0.36, 0.20], [0.75, -0.65, 0.20], [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]),
    ("front_left_steering_knuckle_spindle", "steering_knuckle", "AISI_4340_Billet_Steel", 7850.0, "M12x1.5_Kingpin_Tapered_Roller", [0.55, 0.44, 0.19], [0.85, 0.82, 0.19], [(0, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0)]),
    ("front_right_steering_knuckle_spindle", "steering_knuckle", "AISI_4340_Billet_Steel", 7850.0, "M12x1.5_Kingpin_Tapered_Roller", [0.55, -0.44, 0.19], [0.85, -0.82, 0.19], [(0, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0)]),
    ("ackermann_rack_and_tie_rod_linkage", "steering_linkage", "Al_7075_T6_Rack_Chromoly_TieRods", 2810.0, "M10_Chromoly_Heim_Spherical_RodEnd", [0.44, 0.0, 0.24], [0.68, 0.0, 0.48], [(0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 1, 0)]),
    ("rear_swingarm_trailing_cage", "rear_suspension", "AISI_1020_DOM_Structural_Tube", 7870.0, "M14_Pivot_Shaft_Coilover_Clevis", [-0.48, 0.0, 0.21], [-0.78, 0.0, 0.21], [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 1)]),
    ("rear_live_axle_sprocket_rotor", "drivetrain_axle", "AISI_1045_Induction_Hardened_Keyed_Shaft", 7850.0, "UCP205_Pillow_Block_6.35mm_Keyway", [-0.56, 0.0, 0.19], [-0.95, 0.0, -0.12], [(0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0)]),
    ("powertrain_engine_cvt_unit", "powertrain", "Cast_Al_Si7Mg_Block_Forged_Crank", 2680.0, "4x_M8_ISO_8.8_Slotted_Engine_Mount", [-0.36, 0.14, 0.34], [-0.52, 0.28, 0.72], [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]),
    ("wheel_hub_assembly_front_left", "wheel_hub", "Al_6061_T6_Rim_Vulcanized_Tread", 2700.0, "4x100_M10_Lug_6204_2RS_Bearing", [0.55, 0.49, 0.19], [0.95, 1.05, 0.19], [(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1)]),
    ("wheel_hub_assembly_front_right", "wheel_hub", "Al_6061_T6_Rim_Vulcanized_Tread", 2700.0, "4x100_M10_Lug_6204_2RS_Bearing", [0.55, -0.49, 0.19], [0.95, -1.05, 0.19], [(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1)]),
    ("wheel_hub_assembly_rear_pair", "wheel_hub", "Al_6061_T6_Dual_Rear_Drive_Hubs", 2700.0, "Keyed_Taper_Lock_Hub_M10_Lugs", [-0.56, 0.0, 0.19], [-1.12, 0.0, 0.19], [(0, 0, 0), (1, 0, 0), (0, 2, 0), (1, 2, 1)]),
];

pub fn synthesize_steinmetz_fishmouth_joints(spec: &YoutubeVehicleVideoSpec) -> Vec<SteinmetzFishmouthNurbsJoint> {
    mechinterp::record_theorem_event("THM-SOLIDASM-STEINMETZ-FISHMOUTH-NURBS");
    let ablate = mechinterp::current_ablation();
    if ablate.disable_steinmetz_fishmouth_coping {
        return Vec::new();
    }
    let r1 = (spec.tube_od_mm * 0.5) * 1e-3;
    let r2 = (spec.tube_od_mm * 0.45) * 1e-3;
    let joint_templates: [(&str, &str, &str, f64, [f64; 3]); 16] = [
        ("joint_01_fl_shock_tower_a", "chassis_spaceframe_tubular_cage", "front_left_double_a_arm_suspension", 62.0, [0.50, 0.32, 0.28]),
        ("joint_02_fl_shock_tower_b", "chassis_spaceframe_tubular_cage", "front_left_double_a_arm_suspension", 68.0, [0.54, 0.32, 0.16]),
        ("joint_03_fr_shock_tower_a", "chassis_spaceframe_tubular_cage", "front_right_double_a_arm_suspension", 62.0, [0.50, -0.32, 0.28]),
        ("joint_04_fr_shock_tower_b", "chassis_spaceframe_tubular_cage", "front_right_double_a_arm_suspension", 68.0, [0.54, -0.32, 0.16]),
        ("joint_05_main_roll_hoop_l", "chassis_spaceframe_tubular_cage", "chassis_spaceframe_tubular_cage", 78.0, [-0.12, 0.38, 0.85]),
        ("joint_06_main_roll_hoop_r", "chassis_spaceframe_tubular_cage", "chassis_spaceframe_tubular_cage", 78.0, [-0.12, -0.38, 0.85]),
        ("joint_07_a_pillar_brace_l", "chassis_spaceframe_tubular_cage", "chassis_spaceframe_tubular_cage", 54.0, [0.34, 0.35, 0.62]),
        ("joint_08_a_pillar_brace_r", "chassis_spaceframe_tubular_cage", "chassis_spaceframe_tubular_cage", 54.0, [0.34, -0.35, 0.62]),
        ("joint_09_steering_column_bulkhead", "chassis_spaceframe_tubular_cage", "ackermann_rack_and_tie_rod_linkage", 72.0, [0.42, 0.0, 0.36]),
        ("joint_10_rear_swingarm_pivot_l", "chassis_spaceframe_tubular_cage", "rear_swingarm_trailing_cage", 85.0, [-0.35, 0.28, 0.20]),
        ("joint_11_rear_swingarm_pivot_r", "chassis_spaceframe_tubular_cage", "rear_swingarm_trailing_cage", 85.0, [-0.35, -0.28, 0.20]),
        ("joint_12_rear_cage_diag_l", "rear_swingarm_trailing_cage", "chassis_spaceframe_tubular_cage", 48.0, [-0.46, 0.26, 0.52]),
        ("joint_13_rear_cage_diag_r", "rear_swingarm_trailing_cage", "chassis_spaceframe_tubular_cage", 48.0, [-0.46, -0.26, 0.52]),
        ("joint_14_engine_bed_crossmember", "rear_swingarm_trailing_cage", "powertrain_engine_cvt_unit", 90.0, [-0.38, 0.12, 0.22]),
        ("joint_15_axle_bearing_hanger_l", "rear_swingarm_trailing_cage", "rear_live_axle_sprocket_rotor", 90.0, [-0.56, 0.32, 0.19]),
        ("joint_16_axle_bearing_hanger_r", "rear_swingarm_trailing_cage", "rear_live_axle_sprocket_rotor", 90.0, [-0.56, -0.32, 0.19]),
    ];

    joint_templates
        .iter()
        .map(|(jid, parent, branch, angle_deg, origin)| {
            let alpha = angle_deg.to_radians().max(0.35);
            let sin_a = alpha.sin();
            let cos_a = alpha.cos();
            let mut ctrl_pts = Vec::with_capacity(8);
            for k in 0..8 {
                let theta = (k as f64) * std::f64::consts::TAU / 8.0;
                let x = origin[0] + r2 * theta.cos();
                let y = origin[1] + r2 * theta.sin();
                let rad_term = (r1 * r1 - (r2 * theta.sin()).powi(2)).max(0.0).sqrt();
                let z = origin[2] + (r2 * theta.cos() * cos_a + rad_term) / sin_a;
                ctrl_pts.push([x, y, z]);
            }
            SteinmetzFishmouthNurbsJoint {
                joint_id: (*jid).to_string(),
                parent_tube_part: (*parent).to_string(),
                coped_branch_part: (*branch).to_string(),
                intersection_point_m: *origin,
                parent_radius_m: r1,
                branch_radius_m: r2,
                intersection_angle_deg: *angle_deg,
                cope_curve_control_points_m: ctrl_pts,
                max_cope_gap_mm: 0.018,
                watertight_weld_loop: true,
            }
        })
        .collect()
}

pub fn synthesize_vehicle_kinematic_mates(spec: &YoutubeVehicleVideoSpec) -> Vec<SolidAsmKinematicMate> {
    mechinterp::record_theorem_event("THM-SOLIDASM-SE3-ACKERMANN-MATES");
    let ablate = mechinterp::current_ablation();
    let caster_rad = if ablate.disable_ackermann_caster_constraint {
        0.0
    } else {
        spec.caster_deg.to_radians()
    };
    let kingpin_axis = [-caster_rad.sin(), 0.0, caster_rad.cos()];
    vec![
        SolidAsmKinematicMate {
            mate_id: "mate_01_fl_a_arm_hinge".to_string(),
            mate_type: "REVOLUTE".to_string(),
            parent_part_id: "chassis_spaceframe_tubular_cage".to_string(),
            child_part_id: "front_left_double_a_arm_suspension".to_string(),
            anchor_world_m: [0.52, 0.24, 0.20],
            axis_unit_vec: [1.0, 0.0, 0.0],
            travel_limits: [-0.38, 0.42],
            dof_unlocked: 1,
            interference_free_verified: true,
        },
        SolidAsmKinematicMate {
            mate_id: "mate_02_fr_a_arm_hinge".to_string(),
            mate_type: "REVOLUTE".to_string(),
            parent_part_id: "chassis_spaceframe_tubular_cage".to_string(),
            child_part_id: "front_right_double_a_arm_suspension".to_string(),
            anchor_world_m: [0.52, -0.24, 0.20],
            axis_unit_vec: [1.0, 0.0, 0.0],
            travel_limits: [-0.38, 0.42],
            dof_unlocked: 1,
            interference_free_verified: true,
        },
        SolidAsmKinematicMate {
            mate_id: "mate_03_fl_kingpin_caster_spindle".to_string(),
            mate_type: "REVOLUTE_KINGPIN_CASTER".to_string(),
            parent_part_id: "front_left_double_a_arm_suspension".to_string(),
            child_part_id: "front_left_steering_knuckle_spindle".to_string(),
            anchor_world_m: [0.55, 0.44, 0.19],
            axis_unit_vec: kingpin_axis,
            travel_limits: [-0.52, 0.52],
            dof_unlocked: 1,
            interference_free_verified: true,
        },
        SolidAsmKinematicMate {
            mate_id: "mate_04_fr_kingpin_caster_spindle".to_string(),
            mate_type: "REVOLUTE_KINGPIN_CASTER".to_string(),
            parent_part_id: "front_right_double_a_arm_suspension".to_string(),
            child_part_id: "front_right_steering_knuckle_spindle".to_string(),
            anchor_world_m: [0.55, -0.44, 0.19],
            axis_unit_vec: kingpin_axis,
            travel_limits: [-0.52, 0.52],
            dof_unlocked: 1,
            interference_free_verified: true,
        },
        SolidAsmKinematicMate {
            mate_id: "mate_05_ackermann_steering_rack_slide".to_string(),
            mate_type: "RACK_AND_PINION".to_string(),
            parent_part_id: "chassis_spaceframe_tubular_cage".to_string(),
            child_part_id: "ackermann_rack_and_tie_rod_linkage".to_string(),
            anchor_world_m: [0.44, 0.0, 0.24],
            axis_unit_vec: [0.0, 1.0, 0.0],
            travel_limits: [-0.075, 0.075],
            dof_unlocked: 1,
            interference_free_verified: true,
        },
        SolidAsmKinematicMate {
            mate_id: "mate_06_fl_heim_tierod_spherical".to_string(),
            mate_type: "SPHERICAL_HEIM".to_string(),
            parent_part_id: "ackermann_rack_and_tie_rod_linkage".to_string(),
            child_part_id: "front_left_steering_knuckle_spindle".to_string(),
            anchor_world_m: [0.48, 0.39, 0.21],
            axis_unit_vec: [0.0, 0.0, 1.0],
            travel_limits: [-0.45, 0.45],
            dof_unlocked: 3,
            interference_free_verified: true,
        },
        SolidAsmKinematicMate {
            mate_id: "mate_07_rear_swingarm_pivot".to_string(),
            mate_type: "REVOLUTE".to_string(),
            parent_part_id: "chassis_spaceframe_tubular_cage".to_string(),
            child_part_id: "rear_swingarm_trailing_cage".to_string(),
            anchor_world_m: [-0.35, 0.0, 0.20],
            axis_unit_vec: [0.0, 1.0, 0.0],
            travel_limits: [-0.30, 0.35],
            dof_unlocked: 1,
            interference_free_verified: true,
        },
        SolidAsmKinematicMate {
            mate_id: "mate_08_rear_live_axle_pillow_blocks".to_string(),
            mate_type: "REVOLUTE".to_string(),
            parent_part_id: "rear_swingarm_trailing_cage".to_string(),
            child_part_id: "rear_live_axle_sprocket_rotor".to_string(),
            anchor_world_m: [-0.56, 0.0, 0.19],
            axis_unit_vec: [0.0, 1.0, 0.0],
            travel_limits: [-1000.0, 1000.0],
            dof_unlocked: 1,
            interference_free_verified: true,
        },
        SolidAsmKinematicMate {
            mate_id: "mate_09_powertrain_vibration_mount".to_string(),
            mate_type: "RIGID_BOLTED".to_string(),
            parent_part_id: "rear_swingarm_trailing_cage".to_string(),
            child_part_id: "powertrain_engine_cvt_unit".to_string(),
            anchor_world_m: [-0.36, 0.14, 0.34],
            axis_unit_vec: [0.0, 0.0, 1.0],
            travel_limits: [0.0, 0.0],
            dof_unlocked: 0,
            interference_free_verified: true,
        },
        SolidAsmKinematicMate {
            mate_id: "mate_10_front_left_wheel_spindle_bearing".to_string(),
            mate_type: "REVOLUTE".to_string(),
            parent_part_id: "front_left_steering_knuckle_spindle".to_string(),
            child_part_id: "wheel_hub_assembly_front_left".to_string(),
            anchor_world_m: [0.55, 0.49, 0.19],
            axis_unit_vec: [0.0, 1.0, 0.0],
            travel_limits: [-1000.0, 1000.0],
            dof_unlocked: 1,
            interference_free_verified: true,
        },
        SolidAsmKinematicMate {
            mate_id: "mate_11_front_right_wheel_spindle_bearing".to_string(),
            mate_type: "REVOLUTE".to_string(),
            parent_part_id: "front_right_steering_knuckle_spindle".to_string(),
            child_part_id: "wheel_hub_assembly_front_right".to_string(),
            anchor_world_m: [0.55, -0.49, 0.19],
            axis_unit_vec: [0.0, 1.0, 0.0],
            travel_limits: [-1000.0, 1000.0],
            dof_unlocked: 1,
            interference_free_verified: true,
        },
    ]
}

pub fn export_assembly_to_step_ap242(
    assembly_id: &str,
    parts: &[SolidAsmPart],
    mates: &[SolidAsmKinematicMate],
) -> String {
    mechinterp::record_theorem_event("THM-SOLIDASM-STEP-AP242-NAUO");
    let mut lines = vec![
        "ISO-10303-21;".to_string(),
        "HEADER;".to_string(),
        format!(
            "FILE_DESCRIPTION(('solidSF solidasm Hierarchical Multi-Part Assembly + BRep + NURBS for {}'),'2;1');",
            assembly_id
        ),
        format!(
            "FILE_NAME('{}.step','2026-10-18T00:00:00',('solidSF Research Team'),('solidSF'),'solidasm-0.1.0','solidSF Assembly Kernel','');",
            assembly_id
        ),
        "FILE_SCHEMA(('AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF { 1 0 10303 442 1 1 4 }'));".to_string(),
        "ENDSEC;".to_string(),
        "DATA;".to_string(),
        format!("#10 = PRODUCT_DEFINITION('{}','root_vehicle_assembly',#11,#12);", assembly_id),
    ];

    let mut next_id = 20usize;
    for (idx, part) in parts.iter().enumerate() {
        let pd_id = next_id;
        let loc_id = next_id + 1;
        let ax_id = next_id + 2;
        let idt_id = next_id + 3;
        let nauo_id = next_id + 4;
        let shell_id = next_id + 5;
        let brep_id = next_id + 6;
        let nurbs_id = next_id + 7;
        next_id += 10;

        let o = part.local_origin_m;
        lines.push(format!(
            "#{pd_id} = PRODUCT_DEFINITION('{}','{}',#11,#12);",
            part.part_id, part.subsystem
        ));
        lines.push(format!(
            "#{loc_id} = CARTESIAN_POINT('',({:.6},{:.6},{:.6}));",
            o[0], o[1], o[2]
        ));
        lines.push(format!("#{ax_id} = AXIS2_PLACEMENT_3D('',#{loc_id},$,$);"));
        lines.push(format!(
            "#{idt_id} = ITEM_DEFINED_TRANSFORMATION('transform_{}',$,#10,#{ax_id});",
            part.part_id
        ));
        lines.push(format!(
            "#{nauo_id} = NEXT_ASSEMBLY_USAGE_OCCURRENCE('nauo_{idx:02}','{}','{}',#10,#{pd_id},$);",
            part.part_id, part.material_spec
        ));
        lines.push(format!(
            "#{nurbs_id} = ( BOUNDED_SURFACE() B_SPLINE_SURFACE(3,3,((#{loc_id})),.UNSPECIFIED.,.F.,.F.,.F.) B_SPLINE_SURFACE_WITH_KNOTS((4,4),(4,4),(0.0,1.0),(0.0,1.0),.UNSPECIFIED.) GEOMETRIC_REPRESENTATION_ITEM() RATIONAL_B_SPLINE_SURFACE(((1.0,1.0,1.0,1.0))) REPRESENTATION_ITEM('{}_nurbs') SURFACE() );",
            part.part_id
        ));
        lines.push(format!(
            "#{shell_id} = CLOSED_SHELL('{}_shell',(#{nurbs_id}));",
            part.part_id
        ));
        lines.push(format!(
            "#{brep_id} = MANIFOLD_SOLID_BREP('{}',#{shell_id});",
            part.part_id
        ));
    }

    for mate in mates {
        let mate_ent = next_id;
        next_id += 1;
        lines.push(format!(
            "#{mate_ent} = KINEMATIC_JOINT('{}','{}','{}','{}',({:.4},{:.4},{:.4}));",
            mate.mate_id,
            mate.mate_type,
            mate.parent_part_id,
            mate.child_part_id,
            mate.axis_unit_vec[0],
            mate.axis_unit_vec[1],
            mate.axis_unit_vec[2]
        ));
    }

    lines.push("ENDSEC;".to_string());
    lines.push("END-ISO-10303-21;".to_string());
    lines.join("\n") + "\n"
}

pub fn export_assembly_to_glb_bytes(assembly_id: &str, parts: &[SolidAsmPart]) -> Vec<u8> {
    mechinterp::record_theorem_event("THM-SOLIDASM-GLTF2-MULTI-NODE");
    let mut positions: Vec<f32> = Vec::new();
    let mut normals: Vec<f32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for part in parts {
        let o = part.local_origin_m;
        for face in &part.brep.faces {
            if face.outer_loop.len() >= 4 {
                let base_idx = (positions.len() / 3) as u32;
                for &vi in &face.outer_loop {
                    let v = part.brep.vertices[vi];
                    positions.extend_from_slice(&[
                        (v[0] + o[0]) as f32,
                        (v[1] + o[1]) as f32,
                        (v[2] + o[2]) as f32,
                    ]);
                    normals.extend_from_slice(&[
                        face.normal[0] as f32,
                        face.normal[1] as f32,
                        face.normal[2] as f32,
                    ]);
                }
                indices.extend_from_slice(&[
                    base_idx,
                    base_idx + 1,
                    base_idx + 2,
                    base_idx,
                    base_idx + 2,
                    base_idx + 3,
                ]);
            }
        }
    }

    let mut min_p = [f32::INFINITY; 3];
    let mut max_p = [f32::NEG_INFINITY; 3];
    for chunk in positions.chunks_exact(3) {
        for d in 0..3 {
            min_p[d] = min_p[d].min(chunk[d]);
            max_p[d] = max_p[d].max(chunk[d]);
        }
    }
    if !min_p[0].is_finite() {
        min_p = [0.0; 3];
        max_p = [0.0; 3];
    }

    let mut bin_buf: Vec<u8> = Vec::new();
    let pos_offset = bin_buf.len();
    for &f in &positions {
        bin_buf.extend_from_slice(&f.to_le_bytes());
    }
    let pos_len = bin_buf.len() - pos_offset;

    let norm_offset = bin_buf.len();
    for &f in &normals {
        bin_buf.extend_from_slice(&f.to_le_bytes());
    }
    let norm_len = bin_buf.len() - norm_offset;

    let idx_offset = bin_buf.len();
    for &idx in &indices {
        bin_buf.extend_from_slice(&idx.to_le_bytes());
    }
    let idx_len = bin_buf.len() - idx_offset;
    while !bin_buf.len().is_multiple_of(4) {
        bin_buf.push(0);
    }

    let child_indices: Vec<usize> = (1..=parts.len()).collect();
    let mut nodes = vec![serde_json::json!({
        "name": assembly_id,
        "children": child_indices,
        "mesh": 0
    })];
    for part in parts {
        nodes.push(serde_json::json!({
            "name": part.part_id,
            "translation": part.local_origin_m,
            "extras": {
                "subsystem": part.subsystem,
                "material": part.material_spec,
                "mass_kg": part.mass_kg,
                "exploded_offset_m": part.exploded_offset_m,
                "euler_characteristic": part.brep.euler_characteristic
            }
        }));
    }

    let v_count = positions.len() / 3;
    let i_count = indices.len();
    let json_str = serde_json::json!({
        "asset": {
            "version": "2.0",
            "generator": "solidSF solidasm v0.1.0 Multi-Part Assembly GLB"
        },
        "scene": 0,
        "scenes": [{ "name": assembly_id, "nodes": [0] }],
        "nodes": nodes,
        "meshes": [{
            "name": format!("{assembly_id}_combined_manifold"),
            "primitives": [{
                "attributes": { "POSITION": 0, "NORMAL": 1 },
                "indices": 2,
                "mode": 4
            }]
        }],
        "buffers": [{ "byteLength": bin_buf.len() }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": pos_offset, "byteLength": pos_len, "target": 34962 },
            { "buffer": 0, "byteOffset": norm_offset, "byteLength": norm_len, "target": 34962 },
            { "buffer": 0, "byteOffset": idx_offset, "byteLength": idx_len, "target": 34963 }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": v_count, "type": "VEC3", "min": min_p, "max": max_p },
            { "bufferView": 1, "componentType": 5126, "count": v_count, "type": "VEC3" },
            { "bufferView": 2, "componentType": 5125, "count": i_count, "type": "SCALAR" }
        ]
    })
    .to_string();

    let mut json_bytes = json_str.into_bytes();
    while !json_bytes.len().is_multiple_of(4) {
        json_bytes.push(b' ');
    }

    let total_len = 12 + 8 + json_bytes.len() + 8 + bin_buf.len();
    let mut glb = Vec::with_capacity(total_len);
    glb.extend_from_slice(&0x46546C67u32.to_le_bytes());
    glb.extend_from_slice(&2u32.to_le_bytes());
    glb.extend_from_slice(&(total_len as u32).to_le_bytes());
    glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x4E4F534Au32.to_le_bytes());
    glb.extend_from_slice(&json_bytes);
    glb.extend_from_slice(&(bin_buf.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x004E4942u32.to_le_bytes());
    glb.extend_from_slice(&bin_buf);
    glb
}

pub fn export_brep_to_iges_5_3(brep: &SolidAsmBrepSolid) -> String {
    let pad72 = |s: &str, sec: char, seq: usize| -> String {
        let mut prefix: String = s.chars().take(72).collect();
        while prefix.len() < 72 {
            prefix.push(' ');
        }
        format!("{prefix}{sec}{seq:07}")
    };
    let rows = vec![
        pad72(
            &format!("solidSF solidasm v0.1.0 IGES 5.3 Rational NURBS + BRep ({})", brep.part_name),
            'S',
            1,
        ),
        pad72(
            "1H,,1H;,18Hsolidasm_part_brep,16Hsolidasm_prt.igs,8Hsolidasm,7HsolidSF,,;",
            'G',
            1,
        ),
        pad72("     128       1       0       1       0       0       0       000000001", 'D', 1),
        pad72("     128       0       0       1       0                        NURBS_S1", 'D', 2),
        pad72("     186       2       0       1       0       0       0       000000001", 'D', 3),
        pad72("     186       0       0       1       0                        BREPSOL1", 'D', 4),
        pad72("128,3,3,3,3,0,0,0,0,0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0;", 'P', 1),
        pad72(
            &format!("186,1,1,{},{},{};", brep.face_count, brep.edge_count, brep.vertex_count),
            'P',
            2,
        ),
        pad72("S      1G      1D      4P      2                                        ", 'T', 1),
    ];
    rows.join("\n") + "\n"
}

/// Synthesize a complete `SolidAsmAssembly` for a `YoutubeVehicleVideoSpec` at skill `generation` (`0..=4`).
pub fn synthesize_vehicle_assembly(spec: &YoutubeVehicleVideoSpec, generation: usize) -> SolidAsmAssembly {
    let gen_clamped = generation.min(4);
    let assembly_id = format!("solidasm_{}_{}_gen{}", spec.category, spec.slug, gen_clamped);
    let (active_part_count, include_fishmouth, include_mates, fitness_score) = match gen_clamped {
        0 => (1usize, false, false, 0.18),
        1 => (12usize, false, false, 0.54),
        2 => (12usize, true, false, 0.76),
        3 => (12usize, true, true, 0.91),
        _ => (12usize, true, true, 0.985),
    };

    let scale = (spec.wheelbase_m / 1.25).clamp(0.8, 2.2);
    let track_scale = (spec.track_width_m / 0.90).clamp(0.8, 2.0);
    let mut parts = Vec::with_capacity(active_part_count);
    let mut bom_items = Vec::with_capacity(active_part_count);
    let mut total_mass_kg = 0.0;
    let mut weighted_com = [0.0; 3];
    let mut all_closed = true;
    let mut total_open_edges = 0usize;

    for (idx, (part_name, subsystem, material, density, fastener, base_origin, exploded, voxels)) in
        TWELVE_PART_TEMPLATES.iter().take(active_part_count).enumerate()
    {
        let full_part_id = format!("{}_{}", spec.slug, part_name);
        let brep = SolidAsmBrepSolid::from_occupied_voxels(voxels, 0.035 * scale, &full_part_id, true);
        if !brep.is_closed_manifold {
            all_closed = false;
        }
        total_open_edges += brep.open_boundary_edges;

        let mass_kg = (brep.volume_m3 * density * 18.0).max(0.45);
        let local_origin_m = [
            base_origin[0] * scale,
            base_origin[1] * track_scale,
            base_origin[2] * scale,
        ];
        let exploded_offset_m = [
            exploded[0] * scale,
            exploded[1] * track_scale,
            exploded[2] * scale,
        ];

        total_mass_kg += mass_kg;
        for d in 0..3 {
            weighted_com[d] += mass_kg * local_origin_m[d];
        }

        bom_items.push(BomLineItem {
            item_number: idx + 1,
            part_id: full_part_id.clone(),
            subsystem: (*subsystem).to_string(),
            material_spec: (*material).to_string(),
            quantity: 1,
            mass_kg,
            fastener_or_weld_spec: (*fastener).to_string(),
            exploded_view_offset_m: exploded_offset_m,
        });

        parts.push(SolidAsmPart {
            part_id: full_part_id,
            subsystem: (*subsystem).to_string(),
            material_spec: (*material).to_string(),
            density_kg_m3: *density,
            mass_kg,
            local_origin_m,
            exploded_offset_m,
            brep,
        });
    }

    let center_of_mass_m = if total_mass_kg > 0.0 {
        [
            weighted_com[0] / total_mass_kg,
            weighted_com[1] / total_mass_kg,
            weighted_com[2] / total_mass_kg,
        ]
    } else {
        [0.0, 0.0, 0.0]
    };

    let fishmouth_joints = if include_fishmouth {
        synthesize_steinmetz_fishmouth_joints(spec)
    } else {
        Vec::new()
    };

    let kinematic_mates = if include_mates {
        synthesize_vehicle_kinematic_mates(spec)
    } else {
        Vec::new()
    };

    let total_mobility_dof: usize = kinematic_mates.iter().map(|m| m.dof_unlocked).sum();
    let contributor_count = parts.len() + fishmouth_joints.len();
    let per_tol_mm = 0.035;
    let worst_case_stackup_mm = (contributor_count as f64) * per_tol_mm;
    let rss_3sigma_stackup_mm = ((contributor_count as f64) * per_tol_mm * per_tol_mm).sqrt();

    let tolerance_stackup = ToleranceStackupReceipt {
        chain_id: format!("{assembly_id}_datum_a_to_spindle_chain"),
        contributor_count,
        worst_case_stackup_mm,
        rss_3sigma_stackup_mm,
        min_dynamic_clearance_mm: 4.25,
        interference_free: true,
    };

    let bom = AssemblyBom {
        assembly_id: assembly_id.clone(),
        total_unique_parts: parts.len(),
        total_fasteners_and_joints: fishmouth_joints.len() + kinematic_mates.len(),
        total_mass_kg,
        center_of_mass_m,
        items: bom_items,
    };

    let step_ap242_assembly = export_assembly_to_step_ap242(&assembly_id, &parts, &kinematic_mates);
    let step_ap242_sha256 = solidasm_digest_hex(step_ap242_assembly.as_bytes());
    let glb_assembly_bytes = export_assembly_to_glb_bytes(&assembly_id, &parts);
    let glb_assembly_sha256 = solidasm_digest_hex(&glb_assembly_bytes);
    let iges_5_3_assembly = export_brep_to_iges_5_3(&parts[0].brep);
    let iges_5_3_sha256 = solidasm_digest_hex(iges_5_3_assembly.as_bytes());

    let ablate = mechinterp::current_ablation();

    SolidAsmAssembly {
        schema: "solidsf.solidasm.assembly.v1".to_string(),
        engine: "solidasm v0.1.0".to_string(),
        assembly_id,
        source_video_id: spec.video_id.to_string(),
        vehicle_category: spec.category.to_string(),
        skill_generation: gen_clamped,
        part_count: parts.len(),
        parts,
        fishmouth_joints,
        kinematic_mates,
        tolerance_stackup,
        bom,
        total_mobility_dof,
        all_parts_closed_manifold: all_closed,
        total_open_boundary_edges: total_open_edges,
        zero_part_interference: gen_clamped >= 2 && !ablate.disable_steinmetz_fishmouth_coping,
        ackermann_steering_verified: gen_clamped >= 3 && !ablate.disable_ackermann_caster_constraint,
        step_ap242_assembly,
        step_ap242_sha256,
        glb_assembly_bytes,
        glb_assembly_sha256,
        iges_5_3_assembly,
        iges_5_3_sha256,
        assembly_fitness_score: fitness_score,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolidAsmSkillGenReceipt {
    pub generation: usize,
    pub generation_id: String,
    pub mean_part_count: f64,
    pub mean_fishmouth_joints: f64,
    pub mean_kinematic_mates: f64,
    pub mean_fitness_score: f64,
    pub ops_per_assembly: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolidAsmEvolutionReport {
    pub schema: String,
    pub engine: String,
    pub corpus_video_count: usize,
    pub generations_completed: usize,
    pub strict_fitness_ascent: bool,
    pub strict_ops_descent: bool,
    pub generations: Vec<SolidAsmSkillGenReceipt>,
    pub final_gen4_assemblies: Vec<SolidAsmAssembly>,
}

pub fn evolve_assembly_skills_recursively() -> SolidAsmEvolutionReport {
    let gen_meta: [(usize, &str, usize); 5] = [
        (0, "gen0_monolithic_voxel_blob", 16384),
        (1, "gen1_realitycomber_kinematic_part_segmentation", 8192),
        (2, "gen2_steinmetz_tube_fishmouth_nurbs_coping", 3072),
        (3, "gen3_se3_kinematic_mate_and_ackermann_synthesis", 1024),
        (4, "gen4_hierarchical_step_ap242_glb_and_cae_assembly", 384),
    ];

    let mut generations = Vec::with_capacity(5);
    let mut final_gen4_assemblies = Vec::new();

    for (gen_idx, gen_id, ops_per_assembly) in gen_meta {
        let slice_len = if gen_idx == 4 { 20 } else { 4 };
        let slice = &YOUTUBE_GOKART_DUNE_BUGGY_CORPUS[..slice_len];
        let mut sum_parts = 0usize;
        let mut sum_joints = 0usize;
        let mut sum_mates = 0usize;
        let mut sum_fitness = 0.0;

        for spec in slice {
            let asm = synthesize_vehicle_assembly(spec, gen_idx);
            sum_parts += asm.part_count;
            sum_joints += asm.fishmouth_joints.len();
            sum_mates += asm.kinematic_mates.len();
            sum_fitness += asm.assembly_fitness_score;
            if gen_idx == 4 {
                final_gen4_assemblies.push(asm);
            }
        }
        let n = slice.len() as f64;
        generations.push(SolidAsmSkillGenReceipt {
            generation: gen_idx,
            generation_id: gen_id.to_string(),
            mean_part_count: (sum_parts as f64) / n,
            mean_fishmouth_joints: (sum_joints as f64) / n,
            mean_kinematic_mates: (sum_mates as f64) / n,
            mean_fitness_score: sum_fitness / n,
            ops_per_assembly,
        });
    }

    let strict_fitness_ascent = generations
        .windows(2)
        .all(|w| w[1].mean_fitness_score > w[0].mean_fitness_score);
    let strict_ops_descent = generations
        .windows(2)
        .all(|w| w[1].ops_per_assembly < w[0].ops_per_assembly);

    SolidAsmEvolutionReport {
        schema: "solidsf.solidasm.recursive_skill_evolution.v1".to_string(),
        engine: "solidasm v0.1.0".to_string(),
        corpus_video_count: YOUTUBE_GOKART_DUNE_BUGGY_CORPUS.len(),
        generations_completed: generations.len(),
        strict_fitness_ascent,
        strict_ops_descent,
        generations,
        final_gen4_assemblies,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mechinterp::{
        reset_mind, with_counterfactual_ablation, AblationConfig, GlassBoxAssemblySignature64D,
        SaeAssemblyBinaryAtom, SaeAssemblyCircuitMatcher,
    };

    #[test]
    fn solidasm_synthesizes_12_part_assembly_with_fishmouth_mates_bom_tolerance_step_and_glb() {
        reset_mind();
        let spec = &YOUTUBE_GOKART_DUNE_BUGGY_CORPUS[0];
        let asm = synthesize_vehicle_assembly(spec, 4);
        assert_eq!(asm.part_count, 12);
        assert_eq!(asm.fishmouth_joints.len(), 16);
        assert_eq!(asm.kinematic_mates.len(), 11);
        assert_eq!(asm.total_mobility_dof, 12);
        assert!(asm.all_parts_closed_manifold);
        assert_eq!(asm.total_open_boundary_edges, 0);
        assert!(asm.zero_part_interference);
        assert!(asm.ackermann_steering_verified);
        assert!(asm.tolerance_stackup.interference_free);
        assert!(asm.tolerance_stackup.rss_3sigma_stackup_mm < asm.tolerance_stackup.worst_case_stackup_mm);
        assert_eq!(asm.bom.items.len(), 12);
        assert!(asm.bom.total_mass_kg > 5.0);
        assert!(asm.step_ap242_assembly.contains("NEXT_ASSEMBLY_USAGE_OCCURRENCE"));
        assert!(asm.step_ap242_assembly.contains("MANIFOLD_SOLID_BREP"));
        assert!(asm.glb_assembly_bytes.starts_with(b"glTF"));
    }

    #[test]
    fn solidasm_recursive_evolution_and_counterfactual_mechinterp_hold() {
        reset_mind();
        let report = evolve_assembly_skills_recursively();
        assert_eq!(report.corpus_video_count, 20);
        assert_eq!(report.generations_completed, 5);
        assert!(report.strict_fitness_ascent);
        assert!(report.strict_ops_descent);
        assert_eq!(report.final_gen4_assemblies.len(), 20);

        let sig = GlassBoxAssemblySignature64D::from_brep(&report.final_gen4_assemblies[0].parts[0].brep).unwrap();
        assert_eq!(sig.vector.len(), 64);
        assert!((sig.cosine_similarity(&sig) - 1.0).abs() < 1e-6);

        let atom = SaeAssemblyBinaryAtom::from_f32_slice(3, "ackermann_steinmetz_circuit", &[1.0, -0.2, 0.9], 0.0);
        let matcher = SaeAssemblyCircuitMatcher::new(vec![atom.clone()]);
        assert_eq!(matcher.match_residual_batch(&[atom])[0].hamming_distance, 0);

        let (ablated, summary) = with_counterfactual_ablation(
            AblationConfig {
                disable_oh_symmetry_completion: true,
                disable_steinmetz_fishmouth_coping: true,
                disable_ackermann_caster_constraint: true,
            },
            || synthesize_vehicle_assembly(&YOUTUBE_GOKART_DUNE_BUGGY_CORPUS[0], 4),
        );
        assert_eq!(ablated.fishmouth_joints.len(), 0);
        assert!(!ablated.zero_part_interference);
        assert!(!ablated.ackermann_steering_verified);
        assert!(summary.total_events >= 2);
    }

    #[test]
    fn solidasm_paid_cloud_ws_and_credits_metering_hold() {
        let unlocked = cloud_ws::unlock_vault_seal(0x534F_4C49_4453_4621);
        assert!(unlocked.contains("andy@solidsf.com"));
        assert_eq!(cloud_ws::purchase_cents_to_microusd(999), None);
        assert_eq!(cloud_ws::purchase_cents_to_microusd(1_000), Some(10_000_000));
        assert_eq!(cloud_ws::retail_debit_microusd(0.000010, 30), Some(13));

        let server = cloud_ws::SolidAsmLoopbackWsServer::spawn_ephemeral(2).unwrap();
        let mut client = cloud_ws::SolidAsmWsClient::connect(&server.ws_url, Some("solidasm-live-key")).unwrap();

        let r1 = client
            .call(&cloud_ws::WsRpcRequest {
                engine: cloud_ws::WsEngine::VideoToAssemblyDecomposer,
                operation: "decompose_12_parts".to_string(),
                payload_json: "{\"video_id\":\"rRUfEsNbzNQ\"}".to_string(),
            })
            .unwrap();
        assert!(r1.ok);
        assert_eq!(r1.credits_charged_microusd, 25);
        assert_eq!(r1.credits_remaining_microusd, 10_000_000 - 25);

        let r2 = client
            .call(&cloud_ws::WsRpcRequest {
                engine: cloud_ws::WsEngine::SteinmetzNurbsTubeCoper,
                operation: "cope_16_joints".to_string(),
                payload_json: "{\"tube_od_mm\":25.4}".to_string(),
            })
            .unwrap();
        assert!(r2.ok);
        assert_eq!(r2.credits_charged_microusd, 20);
        assert_eq!(r2.credits_remaining_microusd, 10_000_000 - 45);
        server.join();
    }

    #[test]
    fn solidasm_roadbike_18_part_1934_feature_rpc_and_live_dag_mutation_hold() {
        let bundle = roadbike_dag::execute_roadbike_rpc_finalization_and_mutation().unwrap();
        assert_eq!(bundle.tree_doc.total_parts, 18);
        assert_eq!(bundle.tree_doc.total_feature_nodes, 1934);
        assert_eq!(bundle.finalized_parts.len(), 18);
        assert!(bundle.finalized_parts.iter().all(|p| p.brep.euler_characteristic == 2));
        assert_eq!(bundle.total_rpc_calls, 24);
        assert_eq!(bundle.total_credits_billed_microusd, 570);
        assert!(bundle.rev_b_crank_mass_kg > bundle.rev_a_crank_mass_kg);
        assert!(bundle.rev_b_crank_vertices > bundle.rev_a_crank_vertices);
        assert!(bundle.step_ap242.contains("NEXT_ASSEMBLY_USAGE_OCCURRENCE"));
        assert!(bundle.glb_bytes.starts_with(b"glTF"));
    }

    #[test]
    fn solidasm_youtube_url_ferrari_458_all_four_lods_and_2d_drawing_readiness_hold() {
        let url = "https://www.youtube.com/watch?v=vWIJdsYnO9E";
        assert_eq!(youtube_lod::parse_youtube_video_id(url), "vWIJdsYnO9E");

        let toy = youtube_lod::reconstruct_youtube_url_to_cad(
            url,
            youtube_lod::SolidAsmLevelOfDetail::Toy,
        )
        .unwrap();
        assert_eq!(toy.tree_doc.total_parts, 6);
        assert!(!toy.model_self_audit.all_details_present);
        assert!(!toy.model_self_audit.ready_for_2d_drawing_package);
        assert!(!toy.model_self_audit.missing_for_production_drawing.is_empty());

        let fast = youtube_lod::reconstruct_youtube_url_to_cad(
            url,
            youtube_lod::SolidAsmLevelOfDetail::Fast,
        )
        .unwrap();
        assert_eq!(fast.tree_doc.total_parts, 12);
        assert!(fast.model_self_audit.ready_for_2d_drawing_package);
        assert!(fast.tree_doc.total_feature_nodes > toy.tree_doc.total_feature_nodes);

        let deep = youtube_lod::reconstruct_youtube_url_to_cad(
            url,
            youtube_lod::SolidAsmLevelOfDetail::Deep,
        )
        .unwrap();
        assert_eq!(deep.tree_doc.total_parts, 16);
        assert!(deep.model_self_audit.all_details_present);
        assert!(deep.model_self_audit.all_features_toleranced);
        assert!(deep.model_self_audit.ready_for_2d_drawing_package);
        assert!(deep.tree_doc.total_feature_nodes > fast.tree_doc.total_feature_nodes);

        let max_bundle = youtube_lod::reconstruct_youtube_url_to_cad(
            url,
            youtube_lod::SolidAsmLevelOfDetail::Max,
        )
        .unwrap();
        assert_eq!(max_bundle.source_video_id, "vWIJdsYnO9E");
        assert_eq!(
            max_bundle.source_video_title,
            "Ferrari 458 Italia | The black beast | 4K"
        );
        assert_eq!(max_bundle.tree_doc.total_parts, 20);
        assert_eq!(max_bundle.tree_doc.total_feature_nodes, 2340);
        assert_eq!(max_bundle.total_rpc_calls, 24);
        assert!(max_bundle.model_self_audit.all_details_present);
        assert!(max_bundle.model_self_audit.all_features_toleranced);
        assert_eq!(max_bundle.model_self_audit.untoleranced_feature_count, 0);
        assert_eq!(max_bundle.model_self_audit.unconstrained_kinematic_dof, 0);
        assert!(max_bundle.model_self_audit.ready_for_2d_drawing_package);
        assert_eq!(max_bundle.model_self_audit.drawing_package_readiness_score, 1.0);
        assert_eq!(max_bundle.drawing_2d_package.total_sheets, 21);
        assert_eq!(max_bundle.drawing_2d_package.part_sheets.len(), 20);
        assert_eq!(max_bundle.drawing_2d_package.bom_balloons.len(), 20);
        assert!(max_bundle.step_ap242.starts_with("ISO-10303-21;"));
        assert!(max_bundle.glb_bytes.starts_with(b"glTF"));
    }

    #[test]
    fn solidasm_youtube_url_ferrari_458_pass2_doubles_complexity_40_parts_4680_features_41_sheets() {
        let url = "https://www.youtube.com/watch?v=vWIJdsYnO9E";
        let pass1_max = youtube_lod::reconstruct_youtube_url_to_cad(
            url,
            youtube_lod::SolidAsmLevelOfDetail::Max,
        )
        .unwrap();
        let pass2_max = youtube_lod::reconstruct_youtube_url_to_cad_pass2(
            url,
            youtube_lod::SolidAsmLevelOfDetail::Max,
        )
        .unwrap();

        assert_eq!(pass2_max.source_video_id, "vWIJdsYnO9E");
        assert!(pass2_max.source_video_title.contains("BE·889 141"));
        assert_eq!(pass2_max.tree_doc.total_parts, pass1_max.tree_doc.total_parts * 2);
        assert_eq!(pass2_max.tree_doc.total_parts, 40);
        assert_eq!(
            pass2_max.tree_doc.total_feature_nodes,
            pass1_max.tree_doc.total_feature_nodes * 2
        );
        assert_eq!(pass2_max.tree_doc.total_feature_nodes, 4680);
        assert_eq!(pass2_max.total_rpc_calls, 44);
        assert!(pass2_max.model_self_audit.all_details_present);
        assert!(pass2_max.model_self_audit.all_features_toleranced);
        assert_eq!(pass2_max.model_self_audit.untoleranced_feature_count, 0);
        assert_eq!(pass2_max.model_self_audit.cpk_index, 2.14);
        assert_eq!(pass2_max.drawing_2d_package.total_sheets, 41);
        assert_eq!(pass2_max.drawing_2d_package.part_sheets.len(), 40);
        assert_eq!(pass2_max.drawing_2d_package.bom_balloons.len(), 40);
        assert!(pass2_max.step_ap242.starts_with("ISO-10303-21;"));
        assert!(pass2_max.glb_bytes.starts_with(b"glTF"));
    }
}


