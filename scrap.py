import os
import time
import requests
from bs4 import BeautifulSoup
import random

def scrape_wikipedia(url, output_file, max_size_mb=10, increment=100*1024, max_depth=5, current_depth=0, visited=set()):
    """
    Scrapes text content from a Wikipedia article and its linked articles, 
    randomly selecting the next link to follow.

    Args:
        url: The URL of the starting Wikipedia article.
        output_file: The name of the file to save the scraped content.
        max_size_mb: The maximum size of the output file in megabytes (MB).
        increment: The chunk size to check for file size growth during scraping.
        max_depth: The maximum depth of links to follow.
        current_depth: The current depth of recursion (internal parameter).
        visited: A set to keep track of visited URLs (internal parameter).
    """

    # Create the file if it doesn't exist
    if not os.path.exists(output_file):
        open(output_file, 'w').close()

    # Check if current depth exceeds max depth, URL is visited, or file size exceeds limit
    if current_depth > max_depth or url in visited or os.path.getsize(output_file) >= max_size_mb * 1024 * 1024:
        return

    visited.add(url)

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract text content from paragraph tags and other relevant elements
    text_elements = soup.find_all(['p', 'li', 'h2', 'h3'])  # Include other relevant elements
    text = " ".join([p.get_text() for p in text_elements]).strip()  # Remove leading/trailing whitespace

    # Further clean the text by removing extra spaces
    cleaned_text = ' '.join(text.split())

    # Save cleaned text content
    with open(output_file, 'a', encoding='utf-8') as file:
        file.write(cleaned_text + '\n\n')  # Add a single newline after each article

    # Print progress percentage
    current_file_size = os.path.getsize(output_file)
    progress_percentage = (current_file_size / (max_size_mb * 1024 * 1024)) * 100
    if current_file_size // increment > (current_file_size - len(cleaned_text)) // increment:
        print(f"Scraping progress: {progress_percentage:.2f}%")

    # Find and follow links (improved link filtering and random selection)
    links = []
    for link in soup.find_all('a', href=True):
        href = link.get('href') 
        if href and href.startswith('/wiki/') and ':' not in href and not href.endswith('(') and not href.endswith(')'): 
            links.append(href)

    if links:
        random_link = random.choice(links)
        full_url = 'https://en.wikipedia.org' + random_link
        time.sleep(2)  # Increased delay to be more cautious
        scrape_wikipedia(full_url, output_file, max_size_mb, increment, max_depth, current_depth + 1, visited)

# Starting URL
start_url = 'https://en.wikipedia.org/wiki/Israel%E2%80%93Hamas_war'
output_file = 'wikipedia_scraped_content.txt'

# Call the function with increased max_depth
scrape_wikipedia(start_url, output_file, max_depth=2)

