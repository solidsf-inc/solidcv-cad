#!/usr/bin/env python3
import os
import subprocess
import time
from playwright.sync_api import sync_playwright

BASE_DIR = "/Users/stu/solidsf-scratch/solidcv-cad-roadbike"
HTML_PATH = f"file://{BASE_DIR}/roadbike_studio.html"
MP4_OUT = f"{BASE_DIR}/solidcv_cad_perfect_roadbike_3d.mp4"
FFMPEG_BIN = "/Users/stu/.local/bin/ffmpeg"
CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

TOTAL_FRAMES = 360
FPS = 30

def main():
    t0 = time.time()
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-f", "image2pipe",
        "-vcodec", "png",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "17",
        "-movflags", "+faststart",
        MP4_OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    keyframe_Exports = {
        0: f"{BASE_DIR}/roadbike_scene0_side_profile.png",
        6: f"{BASE_DIR}/roadbike_scene1_profile.png",
        144: f"{BASE_DIR}/roadbike_scene2_drivetrain.png",
        220: f"{BASE_DIR}/roadbike_scene3_cockpit.png",
        315: f"{BASE_DIR}/roadbike_scene4_exploded.png",
    }

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=CHROME_BIN,
            headless=True,
            args=[
                "--use-gl=angle",
                "--enable-webgl",
                "--ignore-gpu-blocklist",
                "--allow-file-access-from-files",
                "--hide-scrollbars",
            ],
        )
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto(HTML_PATH, wait_until="networkidle")
        page.wait_for_timeout(300)

        for f in range(TOTAL_FRAMES):
            page.evaluate(f"window.renderBikeFrame({f}, {TOTAL_FRAMES})")
            png_bytes = page.screenshot(type="png")
            proc.stdin.write(png_bytes)
            if f in keyframe_Exports:
                with open(keyframe_Exports[f], "wb") as kf:
                    kf.write(png_bytes)
            if (f + 1) % 60 == 0 or f == TOTAL_FRAMES - 1:
                print(f"Rendered {f + 1}/{TOTAL_FRAMES} frames ({time.time() - t0:.1f}s)", flush=True)

        browser.close()

    proc.stdin.close()
    proc.wait()
    sz = os.path.getsize(MP4_OUT)
    print(f"COMPLETE: {MP4_OUT} ({sz:,} bytes) in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
