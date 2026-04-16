"""
FakeShield — Quick Start Script
=================================
Simple entry point to demonstrate the detection pipeline.

Usage:
    python run.py                           # Interactive mode (asks for input)
    python run.py path/to/image.jpg         # Analyze a specific image
    python run.py path/to/video.mp4         # Analyze a specific video

This script uses the modular pipeline from:
    models/    -> Model architecture & loading
    utils/     -> Preprocessing & Grad-CAM
    inference/ -> End-to-end pipeline
"""
import sys
import os


def main():
    # Ensure we can import project modules
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from inference.pipeline import FakeShieldPipeline

    print()
    print("  FakeShield -- Deepfake Detection System")
    print("  ----------------------------------------")
    print()

    # Get input from command line or prompt
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        print("  Enter the path to an image or video file:")
        input_path = input("  > ").strip().strip('"').strip("'")

    if not os.path.isfile(input_path):
        print(f"\n  [ERROR] File not found: {input_path}")
        sys.exit(1)

    # Initialize the pipeline
    pipeline = FakeShieldPipeline()

    # Detect file type by extension
    ext = os.path.splitext(input_path)[1].lower()
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

    if ext in video_exts:
        # -- Video Analysis --
        print(f"\n  [VIDEO] {os.path.basename(input_path)}")
        result = pipeline.analyze_video(input_path)

        print(f"\n  +---------- RESULTS ----------+")
        print(f"  |  Verdict:  {result.verdict:<18}|")
        print(f"  |  Fake %:   {result.fake_percentage:<18.1%}|")
        print(f"  |  Frames:   {result.total_frames_analyzed:<18}|")
        print(f"  |  Duration: {result.duration_seconds:<18.1f}|")
        print(f"  +------------------------------+")

    else:
        # -- Image Analysis --
        print(f"\n  [IMAGE] {os.path.basename(input_path)}")
        result = pipeline.analyze_image(input_path)

        if not result.face_detected:
            print("\n  [WARNING] No face detected in the image.")
            print("  Tip: Use a clear, front-facing photo with a visible face.")
            sys.exit(0)

        print(f"\n  +---------- RESULTS ----------+")
        print(f"  |  Verdict:    {result.label:<16}|")
        print(f"  |  Confidence: {result.confidence:<16.1%}|")
        print(f"  |  Face bbox:  {str(result.bbox):<16}|")
        print(f"  +------------------------------+")

        # Save outputs
        saved = pipeline.save_result(result)
        print(f"\n  [+] Outputs saved to ./results/")
        for name, path in saved.items():
            print(f"      - {name}: {path}")

    print()


if __name__ == "__main__":
    main()
