"""
Auto-shutdown script — Monitors training and shuts down laptop when done.

Usage:
    python shutdown_after_training.py

How it works:
    1. Checks if train_CNN.py is running every 30 seconds
    2. Once it detects training has stopped, waits 60 seconds (for file writes)
    3. Shuts down the laptop automatically

Run this in a SEPARATE terminal while training runs in another.
"""
import subprocess
import time
import sys
import os


def is_training_running():
    """Check if train_CNN.py is currently running."""
    try:
        result = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'],
            capture_output=True, text=True
        )
        # Check if any python process has train_CNN in its command line
        wmic = subprocess.run(
            ['wmic', 'process', 'where', "name='python.exe'", 'get', 'commandline'],
            capture_output=True, text=True
        )
        return 'train_CNN' in wmic.stdout
    except Exception:
        return False


def main():
    print("=" * 50)
    print("🖥️  AUTO-SHUTDOWN MONITOR")
    print("=" * 50)
    print("Waiting for train_CNN.py to start...")
    print("(Run this in a separate terminal)\n")

    # Wait for training to start
    while not is_training_running():
        print("⏳ Training not detected yet... checking again in 10s")
        time.sleep(10)

    print("✅ Training detected! Monitoring...")
    print("💤 Will shut down laptop when training completes.\n")

    # Monitor until training stops
    check_interval = 120  # seconds
    while is_training_running():
        current_time = time.strftime("%H:%M:%S")
        print(f"[{current_time}] Training still running... next check in {check_interval}s")
        time.sleep(check_interval)

    print("\n🏁 Training has FINISHED!")
    print("📊 Generating training plots...")

    # Auto-generate plots
    try:
        subprocess.run(
            [sys.executable, 'plot_training.py',
             '--history', './output/ff_effnet_v2/training_history.csv'],
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        print("✅ Plots saved!")
    except Exception as e:
        print(f"⚠️  Plot generation failed: {e}")

    print("⏳ Waiting 60 seconds for file writes to complete...")
    time.sleep(120)

    print("\n🔌 SHUTTING DOWN in 30 seconds...")
    print("   Press Ctrl+C to CANCEL shutdown!\n")

    try:
        time.sleep(30)
        print("💤 Shutting down NOW...")
        os.system("shutdown /s /t 0")
    except KeyboardInterrupt:
        print("\n❌ Shutdown CANCELLED by user.")
        sys.exit(0)


if __name__ == '__main__':
    main()
