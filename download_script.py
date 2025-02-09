from yt_dlp import YoutubeDL
from typing import Optional, Dict, List
import os

#(remove me) # remove this before running

def download_songs(songs: List[str], output_path: Optional[str] = None) -> Dict[str, str]:
    """
    Automatically download songs from YouTube using first search result.
    
    Args:
        songs (List[str]): List of song names to search and download
        output_path (str, optional): Directory to save the songs
    
    Returns:
        Dict[str, str]: Dictionary mapping song names to their download status
    """
    if output_path is None:
        output_path = os.getcwd()
    
    os.makedirs(output_path, exist_ok=True)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'progress_hooks': [progress_hook],
        'quiet': False,
        'no_warnings': False,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'default_search': 'ytsearch',
    }
    
    results = {}
    
    for i, song in enumerate(songs, 1):
        try:
            print(f"\n[{i}/{len(songs)}] Processing: {song}")
            
            search_query = f"{song} original hindi song"
            
            with YoutubeDL(ydl_opts) as ydl:
                try:
                    info = ydl.extract_info(f"ytsearch1:{search_query}", download=True)['entries'][0]
                    video_title = info['title']
                    results[song] = "Successfully downloaded"
                    print(f"Successfully downloaded: {song} as '{video_title}'")
                    
                except Exception as e:
                    print(f"Error downloading {song}: {str(e)}")
                    results[song] = f"Error: {str(e)}"
                
        except Exception as e:
            print(f"Failed to process {song}: {str(e)}")
            results[song] = f"Failed: {str(e)}"
    
    # Print summary
    print("\nDownload Summary:")
    print("-" * 50)
    successful = sum(1 for status in results.values() if "Successfully" in status)
    failed = len(results) - successful
    print(f"Total songs: {len(songs)}")
    print(f"Successfully downloaded: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed songs:")
        for song, status in results.items():
            if "Successfully" not in status:
                print(f"- {song}: {status}")
    
    return results

def progress_hook(d: Dict) -> None:
    """
    Progress hook to display download progress.
    """
    if d['status'] == 'downloading':
        if 'total_bytes' in d:
            total = d['total_bytes']
            downloaded = d['downloaded_bytes']
            percentage = (downloaded / total) * 100
            print(f"Download Progress: {percentage:.1f}% of {total / (1024*1024):.1f} MB", end='\r')
    elif d['status'] == 'finished':
        print("\nDownload completed! Converting to MP3...")

if __name__ == "__main__":
    # Your list of songs
    songs = [
        ''h''
    ]
    
    output_dir = ''
    if not output_dir:
        output_dir = None
    
    try:
        results = download_songs(songs, output_dir)
    except Exception as e:
        print(f"An error occurred: {str(e)}")