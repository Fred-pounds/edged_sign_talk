#!/usr/bin/env python3
"""
SignASL Video Downloader

This script downloads ASL sign videos from SignASL.org using their embed API.
It extracts the video URL from the widget HTML and downloads the MP4 file.

Usage:
    python download_signasl.py <vidref> <word> [output_dir]
    
Example:
    python download_signasl.py xb7mphumry hello ./data/videos
"""

import requests
import sys
import os
import re
from pathlib import Path
from bs4 import BeautifulSoup


def get_video_url_from_widget(vidref, word_hint):
    """
    Fetch the widget HTML and extract the video source URL.
    
    Args:
        vidref (str): The video reference ID from the embed code (data-vidref)
        word_hint (str): The word being signed (used as a hint parameter)
    
    Returns:
        str: Direct URL to the MP4 video file, or None if not found
    """
    widget_url = f"https://embed-api.signasl.org/widgethtml/{vidref}?wordhint={word_hint}"
    
    try:
        print(f"Fetching widget HTML from: {widget_url}")
        response = requests.get(widget_url, timeout=10)
        response.raise_for_status()
        
        # Parse the HTML to find the video source
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for <video> tag and its <source> child
        video_tag = soup.find('video')
        if video_tag:
            source_tag = video_tag.find('source')
            if source_tag and source_tag.get('src'):
                video_url = source_tag['src']
                print(f"Found video URL: {video_url}")
                return video_url
        
        # Alternative: search for .mp4 URLs in the HTML
        mp4_pattern = re.compile(r'https?://[^\s"\'<>]+\.mp4')
        matches = mp4_pattern.findall(response.text)
        if matches:
            video_url = matches[0]
            print(f"Found video URL via regex: {video_url}")
            return video_url
        
        print("No video URL found in widget HTML")
        return None
        
    except requests.RequestException as e:
        print(f"Error fetching widget HTML: {e}")
        return None


def download_video(video_url, output_path):
    """
    Download the video from the given URL to the specified path.
    
    Args:
        video_url (str): Direct URL to the video file
        output_path (str): Path where the video should be saved
    
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        print(f"Downloading video from: {video_url}")
        print(f"Saving to: {output_path}")
        
        # Stream the download to handle large files
        response = requests.get(video_url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))
        
        # Create parent directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress indication
        downloaded = 0
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}%", end='', flush=True)
        
        print(f"\n✓ Video downloaded successfully: {output_path}")
        print(f"  File size: {downloaded / 1024:.1f} KB")
        return True
        
    except requests.RequestException as e:
        print(f"Error downloading video: {e}")
        return False


def main():
    """Main function to handle command-line usage."""
    if len(sys.argv) < 3:
        print("Usage: python download_signasl.py <vidref> <word> [output_dir]")
        print("\nExample:")
        print("  python download_signasl.py xb7mphumry hello ./data/videos")
        print("\nTo find the vidref:")
        print("  Look for 'data-vidref' in the embed code from SignASL.org")
        sys.exit(1)
    
    vidref = sys.argv[1]
    word = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "./data/videos"
    
    # Construct output filename
    output_path = os.path.join(output_dir, f"{word}.mp4")
    
    # Get the video URL
    video_url = get_video_url_from_widget(vidref, word)
    
    if not video_url:
        print("Failed to extract video URL")
        sys.exit(1)
    
    # Download the video
    success = download_video(video_url, output_path)
    
    if success:
        print(f"\n✓ Success! Video saved to: {output_path}")
        sys.exit(0)
    else:
        print("\n✗ Download failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
