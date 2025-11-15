# data/download_lichess.py
"""
Download Lichess chess databases
September 2025
"""

import subprocess
import os
from pathlib import Path

def download_lichess_month(year, month, output_dir="lichess_data"):
    """Download a specific month's database"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    filename = f"lichess_db_standard_rated_{year}-{month:02d}.pgn.zst"
    url = f"https://database.lichess.org/standard/{filename}"
    output_path = output_dir / filename
    
    if output_path.exists():
        print(f"✓ {filename} already exists")
        return output_path
    
    print(f"Downloading {filename}...")
    print(f"Size: ~8 GB compressed")
    print(f"URL: {url}")
    print("This will take 30-60 minutes...")
    print("=" * 60)
    
    try:
        # Try wget first
        subprocess.run(
            ["wget", "-c", url, "-O", str(output_path)],
            check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to curl
        try:
            subprocess.run(
                ["curl", "-L", "-C", "-", "-o", str(output_path), url],
                check=True
            )
        except subprocess.CalledProcessError:
            print("❌ Download failed. Install wget or curl.")
            return None
    
    size_gb = output_path.stat().st_size / (1024**3)
    print(f"✓ Downloaded: {size_gb:.1f} GB")
    return output_path


if __name__ == "__main__":
    print("=" * 60)
    print("DOWNLOADING LICHESS DATA")
    print("=" * 60)
    
    
    # Download September 2025
    print("\n📥 Downloading September 2024...")
    sep_file = download_lichess_month(2025, 9)
    
    print("\n" + "=" * 60)
    print("✅ DOWNLOAD COMPLETE")
    print("=" * 60)
    print("\nFiles downloaded:")
    # if oct_file:
    #     print(f"  • {oct_file}")
    if sep_file:
        print(f"  • {sep_file}")
    # print("\nNext: Run create_dataset.py")