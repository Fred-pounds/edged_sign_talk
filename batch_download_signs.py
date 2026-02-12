#!/usr/bin/env python3
"""
Batch SignASL Video Downloader

Downloads multiple ASL sign videos from SignASL.org for training purposes.
You need to manually collect the vidref IDs from the SignASL embed codes.

Usage:
    python batch_download_signs.py [output_dir]
"""

import sys
import os
from download_signasl import get_video_url_from_widget, download_video


# List of signs to download: [(word, vidref), ...]
# To add more signs:
# 1. Go to https://www.signasl.org/sign/<word>
# 2. Click "Embed this video"
# 3. Copy the data-vidref value from the embed code
# 4. Add it to this list as a tuple: ("word", "vidref")

SIGNS_TO_DOWNLOAD = [
    ("hello", "xb7mphumry"),
    ("hello", "x6jycjzxdt"),
    ("hello", "mtaohyhunw"),
    ("hello", "ucxom6jtam"),
    ("hello", "ns3nvo9mpj"),
    ("hello", "wglgrqpoan"),
    ("hello", "ptfsc4xmwx"),
    ("help", "n0gkgpn8tf"),
    ("help", "na5bydxogk"),
    ("help", "rumv9iasi0"),
    ("help", "2jgyqopnip"),
    ("help", "mz6rghzuus"),
    ("help", "fkm8uww3ci"),
    ("help", "ku1xzak1wf"),
    ("help", "hmvmo5ukbv"),
    ("help", "h7ewuyge3a"),
    ("help", "shpdlzz5rp"),
    ("help", "ikybu2imrv"),
    ("help", "lxu7t1htw6"),
    ("help", "ypytnllpip"),
    ("help", "nzzurgfxnj"),
    # no (4 videos)
    ("no", "0vycdillx8"),
    ("no", "bxuwdvjaui"),
    ("no", "wpnbu8pmmq"),
    ("no", "ofepcuezwf"),
    # thanks (4 videos)
    ("thanks", "pdmmfefdff"),
    ("thanks", "8iib2gycig"),
    ("thanks", "bch3fvqmtj"),
    ("thanks", "gmbmyyj9u7"),
    # yes (5 videos)
    ("yes", "0vzk33xp6j"),
    ("yes", "cv1mygnmpr"),
    ("yes", "pqgak01sun"),
    ("yes", "gzqucog36n"),
    ("yes", "41n0x9w664"),
]


def batch_download(output_dir="./data/raw"):
    """
    Download all signs defined in SIGNS_TO_DOWNLOAD.
    Videos are organized into subdirectories by sign word: data/raw/<word>/
    
    Args:
        output_dir (str): Directory to save videos
    """
    total = len(SIGNS_TO_DOWNLOAD)
    successful = 0
    failed = []
    
    # Count occurrences of each word to add numbering
    word_counts = {}
    
    print(f"Starting batch download of {total} videos...")
    print(f"Output directory: {output_dir}\n")
    
    for idx, (word, vidref) in enumerate(SIGNS_TO_DOWNLOAD, 1):
        # Track how many times we've seen this word
        word_counts[word] = word_counts.get(word, 0) + 1
        video_num = word_counts[word]
        
        print(f"\n[{idx}/{total}] Processing: {word} (video #{video_num})")
        print("-" * 50)
        
        # Get video URL
        video_url = get_video_url_from_widget(vidref, word)
        
        if not video_url:
            print(f"✗ Failed to get video URL for '{word}'")
            failed.append(f"{word}_{video_num}")
            continue
        
        # Download video with numbering for multiple instances
        # Organize by word: data/raw/<word>/hello_1.mp4, hello_2.mp4, etc.
        word_dir = os.path.join(output_dir, word.replace(' ', '_'))
        filename = f"{word.replace(' ', '_')}_{video_num}.mp4"
        output_path = os.path.join(word_dir, filename)
        success = download_video(video_url, output_path)
        
        if success:
            successful += 1
        else:
            failed.append(f"{word}_{video_num}")
    
    # Summary
    print("\n" + "=" * 50)
    print("BATCH DOWNLOAD SUMMARY")
    print("=" * 50)
    print(f"Total videos: {total}")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {len(failed)}")
    
    if failed:
        print(f"\nFailed downloads: {', '.join(failed)}")
    
    print(f"\nVideos saved to: {output_dir}")


def main():
    """Main function."""
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/videos"
    
    if not SIGNS_TO_DOWNLOAD:
        print("Error: No signs configured in SIGNS_TO_DOWNLOAD dictionary")
        print("\nPlease edit this file and add sign words with their vidref IDs.")
        print("See the comments in the file for instructions.")
        sys.exit(1)
    
    batch_download(output_dir)


if __name__ == "__main__":
    main()
