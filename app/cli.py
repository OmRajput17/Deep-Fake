"""
FakeShield — Command Line Interface
=====================================
Analyze images and videos for deepfake content from the terminal.

Usage:
    # Analyze a single image
    python -m app.cli --image photo.jpg

    # Analyze a video
    python -m app.cli --video clip.mp4

    # Analyze an image without Grad-CAM (faster)
    python -m app.cli --image photo.jpg --no-gradcam

    # Use a custom model checkpoint
    python -m app.cli --image photo.jpg --model ./output/ff_effnet_v1/best.pth

    # Save outputs (face crop, heatmap, overlay) to disk
    python -m app.cli --image photo.jpg --save --output-dir ./results
"""
import argparse
import sys
import os
import time

# Add project root to path so imports work when running as module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.pipeline import FakeShieldPipeline


def print_banner():
    """Print the FakeShield ASCII banner."""
    print()
    print("  +==========================================+")
    print("  |  FakeShield -- Deepfake Detector          |")
    print("  |  EfficientNet-B4 + MTCNN + Grad-CAM       |")
    print("  +==========================================+")
    print()


def analyze_image(args):
    """Run image analysis and print results."""
    pipeline = FakeShieldPipeline(
        model_path=args.model,
        enable_gradcam=not args.no_gradcam,
    )

    print(f"\n[*] Analyzing: {args.image}")
    start = time.time()

    result = pipeline.analyze_image(args.image)

    elapsed = time.time() - start

    print()
    print("  +-------------------------------------+")

    if not result.face_detected:
        print("  |  [!] No face detected in image      |")
        print("  +-------------------------------------+")
        return

    # Verdict
    if result.label == "FAKE":
        print(f"  |  [FAKE]  Verdict: FAKE               |")
    else:
        print(f"  |  [REAL]  Verdict: REAL               |")

    print(f"  |  Confidence: {result.confidence:>6.1%}                |")
    print(f"  |  Inference:  {elapsed*1000:>6.0f}ms                |")
    print(f"  |  Face bbox:  {result.bbox}  |")
    print("  +-------------------------------------+")

    # Save outputs if requested
    if args.save:
        saved = pipeline.save_result(result, args.output_dir)
        print(f"\n  [+] Results saved to {args.output_dir}/")
        for name, path in saved.items():
            print(f"      - {name}: {path}")

    print()


def analyze_video(args):
    """Run video analysis and print results."""
    pipeline = FakeShieldPipeline(
        model_path=args.model,
        enable_gradcam=not args.no_gradcam,
    )

    print(f"\n[*] Analyzing: {args.video}")
    start = time.time()

    result = pipeline.analyze_video(args.video)

    elapsed = time.time() - start

    print()
    print("  +-----------------------------------------+")

    if result.verdict == "FAKE":
        print(f"  |  [FAKE]  Verdict:  FAKE                  |")
    else:
        print(f"  |  [REAL]  Verdict:  REAL                  |")

    print(f"  |  Fake frames:  {result.fake_percentage:>6.1%}                 |")
    print(f"  |  Analyzed:     {result.total_frames_analyzed:>5} frames             |")
    print(f"  |  Real:         {result.real_count:>5} frames             |")
    print(f"  |  Fake:         {result.fake_count:>5} frames             |")
    print(f"  |  Duration:     {result.duration_seconds:>5.1f}s                 |")
    print(f"  |  Processing:   {elapsed:>5.1f}s                 |")
    print("  +-----------------------------------------+")
    print()


def main():
    """CLI entry point."""
    print_banner()

    parser = argparse.ArgumentParser(
        prog="fakeshield",
        description="FakeShield -- AI-Based Deepfake Detection System",
    )

    # Input (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image", "-i",
        help="Path to an image file to analyze",
    )
    input_group.add_argument(
        "--video", "-v",
        help="Path to a video file to analyze",
    )

    # Model config
    parser.add_argument(
        "--model", "-m",
        default="./output/ff_effnet_v1/best.pth",
        help="Path to trained model weights (default: ./output/ff_effnet_v1/best.pth)",
    )
    parser.add_argument(
        "--no-gradcam",
        action="store_true",
        help="Disable Grad-CAM heatmap generation (faster inference)",
    )

    # Output
    parser.add_argument(
        "--save", "-s",
        action="store_true",
        help="Save face crop, heatmap, and overlay to disk",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./results",
        help="Directory to save outputs (default: ./results)",
    )

    args = parser.parse_args()

    if args.image:
        analyze_image(args)
    elif args.video:
        analyze_video(args)


if __name__ == "__main__":
    main()
