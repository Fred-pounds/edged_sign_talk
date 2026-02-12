#!/usr/bin/env python3
"""
SignASL Video Reference Scraper

This script scrapes SignASL.org to extract all video reference IDs (vidref)
for specified sign words. It then updates batch_download_signs.py with the
collected vidrefs.

Usage:
    python scrape_signasl_vidrefs.py <word1> <word2> <word3> ...
    
Example:
    python scrape_signasl_vidrefs.py yes no thanks
"""

import requests
import sys
import re
from bs4 import BeautifulSoup


def scrape_vidrefs_for_word(word):
    """
    Scrape all video reference IDs for a given sign word from SignASL.org.
    
    Args:
        word (str): The sign word to search for
    
    Returns:
        list: List of vidref IDs found for this word
    """
    url = f"https://www.signasl.org/sign/{word}"
    
    try:
        print(f"\nScraping: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all elements with data-vidref attribute
        # These are typically in <blockquote> tags with class "signasldata-embed"
        vidrefs = []
        
        # Method 1: Look for blockquote elements with data-vidref
        for element in soup.find_all(attrs={"data-vidref": True}):
            vidref = element.get('data-vidref')
            if vidref and vidref not in vidrefs:
                vidrefs.append(vidref)
        
        # Method 2: Regex search in the HTML for data-vidref patterns
        if not vidrefs:
            pattern = re.compile(r'data-vidref="([a-z0-9]+)"')
            matches = pattern.findall(response.text)
            vidrefs = list(set(matches))  # Remove duplicates
        
        print(f"  Found {len(vidrefs)} video(s) for '{word}'")
        return vidrefs
        
    except requests.RequestException as e:
        print(f"  Error scraping '{word}': {e}")
        return []


def format_as_tuples(word, vidrefs):
    """
    Format the word and vidrefs as Python tuple strings.
    
    Args:
        word (str): The sign word
        vidrefs (list): List of vidref IDs
    
    Returns:
        list: List of formatted tuple strings
    """
    return [f'    ("{word}", "{vidref}"),' for vidref in vidrefs]


def main():
    """Main function to scrape vidrefs for multiple words."""
    if len(sys.argv) < 2:
        print("Usage: python scrape_signasl_vidrefs.py <word1> <word2> <word3> ...")
        print("\nExample:")
        print("  python scrape_signasl_vidrefs.py yes no thanks")
        sys.exit(1)
    
    words = sys.argv[1:]
    
    print("=" * 60)
    print("SignASL Video Reference Scraper")
    print("=" * 60)
    print(f"Scraping vidrefs for: {', '.join(words)}")
    
    all_results = {}
    
    # Scrape each word
    for word in words:
        vidrefs = scrape_vidrefs_for_word(word)
        if vidrefs:
            all_results[word] = vidrefs
    
    # Display results
    print("\n" + "=" * 60)
    print("SCRAPING RESULTS")
    print("=" * 60)
    
    if not all_results:
        print("No vidrefs found for any word.")
        sys.exit(1)
    
    print("\nAdd these lines to SIGNS_TO_DOWNLOAD in batch_download_signs.py:\n")
    
    for word, vidrefs in all_results.items():
        print(f"    # {word} ({len(vidrefs)} videos)")
        for line in format_as_tuples(word, vidrefs):
            print(line)
    
    print("\n" + "=" * 60)
    print(f"Total: {sum(len(v) for v in all_results.values())} videos across {len(all_results)} words")
    print("=" * 60)


if __name__ == "__main__":
    main()
