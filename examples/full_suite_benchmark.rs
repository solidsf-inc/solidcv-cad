use solidcv_cad::{
    evolve_youtube_assembly_skills_recursively, export_brep_to_iges_5_3, export_cad_to_cae_handoff,
    reconstruct_video_to_glb_and_brep, run_10_clone_solidcv_cad_suite, Mat, CV_8UC1,
};

fn main() {
    let mut f0 = Mat::zeros(16, 16, CV_8UC1).unwrap();
    let mut f1 = Mat::zeros(16, 16, CV_8UC1).unwrap();
    for r in 4..11 {
        for c in 4..11 {
            f0.set_f64(r, c, 0, 210.0);
            f1.set_f64(r + 1, c + 1, 0, 210.0);
        }
    }
    let rec = reconstruct_video_to_glb_and_brep(&[f0, f1], 100, 0.01, "benchmark_turret").unwrap();
    let iges = export_brep_to_iges_5_3(&rec.brep);
    let handoff = export_cad_to_cae_handoff(&rec);
    let suite = run_10_clone_solidcv_cad_suite();
    let yt = evolve_youtube_assembly_skills_recursively();
    println!(
        "solidcv-cad benchmark: closed={} chi={} glb={}B step={}B iges={}B tetra={} 10x_clones={} youtube_videos={} gen4_parts={} fishmouth_joints={} mates={} fitness_ascent={} ops_descent={}",
        rec.brep.is_closed_manifold,
        rec.brep.euler_characteristic,
        rec.glb_bytes.len(),
        rec.step_ap242.len(),
        iges.len(),
        handoff.tetrahedra.len(),
        suite.clone_count,
        yt.corpus_video_count,
        yt.generations[4].mean_part_count,
        yt.generations[4].mean_fishmouth_joints,
        yt.generations[4].mean_kinematic_mates,
        yt.strict_fitness_ascent,
        yt.strict_ops_descent
    );
}

