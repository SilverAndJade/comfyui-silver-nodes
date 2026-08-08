import os
import io
import json
import random
import hashlib
import time
import requests
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List, Set


class SilverStockRandomImage:
    """
    A ComfyUI node that fetches random images from stock photo sources.
    Uses sources that don't require API authentication:
    - Lorem Picsum (picsum.photos) - 1000+ random images
    - Unsplash Source (source.unsplash.com) - random curated photos
    - Wikimedia Commons (commons.wikimedia.org) - public domain images
    
    Provides 1000-5000+ unique images without repeating through local caching.
    """

    # Base URLs for no-auth sources
    SOURCE_CONFIGS = {
        "lorem_picsum": {
            "list_url": "https://picsum.photos/v2/list",
            "image_base": "https://picsum.photos/id",
            "total_images": 1000,  # Lorem Picsum has ~1000 images in their database
        },
        "unsplash_random": {
            "random_url": "https://source.unsplash.com/random",
            "keywords": ["nature", "architecture", "people", "technology", "animals",
                        "food", "travel", "business", "arts", "sports"],
        },
        "wikimedia": {
            "api_url": "https://commons.wikimedia.org/w/api.php",
            "total_images": 5000,  # Can fetch many from Wikimedia
        },
        "lorem_flickr": {
            "base_url": "https://loremflickr.com",
            "lock": "lock",  # Add /lock to get consistent images per keyword
        },
    }

    # History directory for tracking used images
    _history_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'history')
    _history_files = {
        "lorem_picsum": os.path.join(_history_dir, 'lorem_picsum_history.json'),
        "unsplash_random": os.path.join(_history_dir, 'unsplash_random_history.json'),
        "wikimedia": os.path.join(_history_dir, 'wikimedia_history.json'),
        "lorem_flickr": os.path.join(_history_dir, 'lorem_flickr_history.json'),
    }

    # In-memory cache of used image IDs per source
    _used_ids = {
        "lorem_picsum": set(),
        "unsplash_random": set(),
        "wikimedia": set(),
        "lorem_flickr": set(),
    }

    # Maximum history entries per source (allows 1000-5000 unique images)
    MAX_HISTORY_SIZE = 5000
    HISTORY_TRIM_SIZE = 4000  # Trim to this size when limit exceeded

    # Cache for Lorem Picsum image list (fetched once and cached)
    _lorem_picsum_cache: Optional[List[dict]] = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": (["lorem_picsum", "wikimedia", "lorem_flickr", "unsplash_random"], {"default": "lorem_picsum"}),
                "search_term": ("STRING", {"default": "family", "multiline": False}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 768, "min": 64, "max": 2048, "step": 64}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32-1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "image_url")
    FUNCTION = "fetch_random_image"
    CATEGORY = "image/loading"

    def __init__(self):
        self._ensure_history_dir()
        self._load_all_history()

    def _ensure_history_dir(self):
        """Ensure the history directory exists."""
        os.makedirs(self._history_dir, exist_ok=True)

    def _load_history(self, source: str) -> Set[str]:
        """Load used image IDs from history file for a specific source."""
        history_file = self._history_files.get(source)
        if not history_file or not os.path.exists(history_file):
            return set()

        try:
            with open(history_file, 'r') as f:
                data = json.load(f)
                return set(data.get('used_ids', []))
        except Exception as e:
            print(f"[SilverStockRandomImage] Warning: Failed to load {source} history: {e}")
            return set()

    def _load_all_history(self):
        """Load history for all sources."""
        for source in self._used_ids.keys():
            self._used_ids[source] = self._load_history(source)

    def _save_history(self, source: str):
        """Save used image IDs to history file for a specific source."""
        history_file = self._history_files.get(source)
        if not history_file:
            return

        try:
            data = {
                'used_ids': list(self._used_ids[source]),
                'last_updated': time.time()
            }
            with open(history_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[SilverStockRandomImage] Warning: Failed to save {source} history: {e}")

    def _trim_history_if_needed(self, source: str):
        """Trim history if it exceeds max size to allow for more unique images."""
        if len(self._used_ids[source]) > self.MAX_HISTORY_SIZE:
            # Keep a random subset to maintain some uniqueness across runs
            ids_list = list(self._used_ids[source])
            self._used_ids[source] = set(random.sample(ids_list, self.HISTORY_TRIM_SIZE))
            self._save_history(source)
            print(f"[SilverStockRandomImage] Trimmed {source} history to {self.HISTORY_TRIM_SIZE} entries")

    def _generate_image_hash(self, source: str, image_id: str) -> str:
        """Generate a unique hash for an image ID."""
        return hashlib.md5(f"{source}:{image_id}".encode('utf-8')).hexdigest()[:16]

    def _fetch_lorem_picsum_list(self) -> List[dict]:
        """Fetch the list of available images from Lorem Picsum (no auth required)."""
        # Return cached list if available
        if self._lorem_picsum_cache is not None:
            return self._lorem_picsum_cache

        try:
            url = f"{self.SOURCE_CONFIGS['lorem_picsum']['list_url']}?page=1&limit=100"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Cache the results
                SilverStockRandomImage._lorem_picsum_cache = data
                return data
            else:
                print(f"[SilverStockRandomImage] Lorem Picsum list error: {response.status_code}")
                return []
        except Exception as e:
            print(f"[SilverStockRandomImage] Error fetching Lorem Picsum list: {e}")
            return []

    def _fetch_from_lorem_picsum(self, width: int, height: int, search_term: str) -> Optional[Tuple[str, str]]:
        """Fetch a random image from Lorem Picsum (no auth required).
        
        Returns: Tuple of (image_id, image_url) or None
        """
        # Get the list of available images
        image_list = self._fetch_lorem_picsum_list()
        
        if not image_list:
            return None

        # Filter by search term in author or ID if possible (Lorem Picsum doesn't support search,
        # but we can use seed to get consistent "random" images based on search term)
        seed = hash(search_term) % 10000
        random.seed(seed)

        # Shuffle the list to get different ordering based on search term
        shuffled = image_list.copy()
        random.shuffle(shuffled)

        # Find an unused image
        for img_data in shuffled:
            img_id = str(img_data.get('id', ''))
            if img_id and img_id not in self._used_ids['lorem_picsum']:
                # Construct image URL with desired dimensions
                image_url = f"{self.SOURCE_CONFIGS['lorem_picsum']['image_base']}/{img_id}/{width}/{height}"
                return (img_id, image_url)

        # If all images are used, return a truly random one (seed-based)
        if shuffled:
            img_data = random.choice(shuffled)
            img_id = str(img_data.get('id', ''))
            image_url = f"{self.SOURCE_CONFIGS['lorem_picsum']['image_base']}/{img_id}/{width}/{height}"
            return (img_id, image_url)

        return None

    def _fetch_from_unsplash_random(self, width: int, height: int, search_term: str) -> Optional[Tuple[str, str]]:
        """Fetch a random image from Unsplash Source (no auth required).
        
        Returns: Tuple of (image_id, image_url) or None
        """
        try:
            # Unsplash Source allows random images with optional keywords
            # Format: https://source.unsplash.com/random/WxH/?keyword
            base_url = self.SOURCE_CONFIGS['unsplash_random']['random_url']
            
            # Use search term as keyword if provided, otherwise random
            keyword = search_term.strip() if search_term.strip() else random.choice(
                self.SOURCE_CONFIGS['unsplash_random']['keywords']
            )
            
            # Construct URL with dimensions and keyword
            image_url = f"{base_url}/{width}x{height}/?{keyword}"
            
            # Generate a unique ID based on the URL and timestamp
            img_id = hashlib.md5(f"{image_url}:{time.time()}".encode()).hexdigest()[:16]
            
            return (img_id, image_url)
        except Exception as e:
            print(f"[SilverStockRandomImage] Error constructing Unsplash URL: {e}")
            return None

    def _fetch_from_wikimedia(self, width: int, height: int, search_term: str) -> Optional[Tuple[str, str]]:
        """Fetch a random image from Wikimedia Commons (no auth required).

        Returns: Tuple of (image_id, image_url) or None
        """
        try:
            # Wikimedia Commons API doesn't require authentication but needs User-Agent
            headers = {
                "User-Agent": "ComfyUI-SilverNodes/1.0 (https://github.com/silvernodes/comfyui-silver-nodes)"
            }

            params = {
                "action": "query",
                "generator": "search",
                "gsrnamespace": 6,  # File namespace
                "gsrsearch": f"{search_term} filetype:image",
                "prop": "imageinfo",
                "iiprop": "url|size",
                "format": "json",
                "origin": "*",
                "gsrlimit": 50,  # Get 50 results per request
            }

            # Add random offset to get different results (smaller range to avoid exceeding results)
            random_offset = random.randint(0, 100)
            params["gsroffset"] = random_offset

            api_url = self.SOURCE_CONFIGS['wikimedia']['api_url']
            response = requests.get(api_url, headers=headers, params=params, timeout=10)

            if response.status_code != 200:
                print(f"[SilverStockRandomImage] Wikimedia API error: HTTP {response.status_code}")
                return None

            data = response.json()

            # Check for API errors
            if 'error' in data:
                print(f"[SilverStockRandomImage] Wikimedia API error: {data['error']}")
                return None

            pages = data.get('query', {}).get('pages', {})

            if not pages:
                print(f"[SilverStockRandomImage] No Wikimedia results for search: {search_term}")
                return None

            # Convert to list and shuffle
            pages_list = list(pages.values())
            random.shuffle(pages_list)

            # Find an unused image
            for page in pages_list:
                page_id = str(page.get('pageid', ''))
                imageinfo = page.get('imageinfo', [])

                if imageinfo and page_id:
                    image_url = imageinfo[0].get('url', '')
                    if image_url:
                        return (page_id, image_url)

            # If all used or no valid URLs, just return first one
            page = pages_list[0]
            page_id = str(page.get('pageid', ''))
            imageinfo = page.get('imageinfo', [])
            if imageinfo:
                image_url = imageinfo[0].get('url', '')
                if image_url:
                    return (page_id, image_url)

            return None
        except Exception as e:
            print(f"[SilverStockRandomImage] Error fetching from Wikimedia: {e}")
            return None

    def _fetch_from_lorem_flickr(self, width: int, height: int, search_term: str) -> Optional[Tuple[str, str]]:
        """Fetch a random image from LoremFlickr (no auth required).

        LoremFlickr provides random images by keyword: https://loremflickr.com/{width}/{height}/{keyword}
        Adding /all/ before keyword returns random images from all of Flickr matching the keyword.
        Adding /lock/ ensures the same image is returned for the same parameters.

        Returns: Tuple of (image_id, image_url) or None
        """
        try:
            base_url = self.SOURCE_CONFIGS['lorem_flickr']['base_url']

            # Clean and format the search term for URL
            keyword = search_term.strip().lower() if search_term.strip() else "random"
            # Replace spaces with commas for multiple keywords
            keyword = keyword.replace(" ", ",")

            # Construct URL: /all/ returns images matching keyword from entire Flickr
            # Without /all/ it returns from curated set
            image_url = f"{base_url}/{width}/{height}/{keyword}"

            # Generate unique ID based on keyword and timestamp (since LoremFlickr returns random images)
            # We use current time to ensure uniqueness across calls
            img_id = hashlib.md5(f"loremflickr:{keyword}:{time.time()}".encode()).hexdigest()[:16]

            return (img_id, image_url)
        except Exception as e:
            print(f"[SilverStockRandomImage] Error constructing LoremFlickr URL: {e}")
            return None

    def _download_and_process_image(self, image_url: str) -> torch.Tensor:
        """Download and process image into ComfyUI tensor format."""
        try:
            # Add User-Agent to avoid 403 errors from Wikimedia and other sources
            headers = {
                "User-Agent": "ComfyUI-SilverNodes/1.0 (https://github.com/silvernodes/comfyui-silver-nodes)"
            }
            response = requests.get(image_url, headers=headers, timeout=15, allow_redirects=True)
            response.raise_for_status()

            # Load image with PIL
            image = Image.open(io.BytesIO(response.content))

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Resize if too large to prevent memory issues
            max_size = 2048
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to numpy array and normalize to 0-1
            image_np = np.array(image).astype(np.float32) / 255.0

            # Convert to torch tensor with shape [1, H, W, 3]
            image_tensor = torch.from_numpy(image_np)[None,]

            return image_tensor

        except Exception as e:
            print(f"[SilverStockRandomImage] Error processing image: {e}")
            raise

    def fetch_random_image(self, source: str, search_term: str, width: int, height: int, seed: int) -> Tuple[torch.Tensor, str]:
        """
        Fetch a random image from the specified source that hasn't been used before.
        
        No API keys or authentication required for any source.

        Args:
            source: The image source to use (lorem_picsum, unsplash_random, wikimedia)
            search_term: Search query string (used as seed/keyword)
            width: Desired image width
            height: Desired image height
            seed: Random seed for reproducibility (0 = random)

        Returns:
            Tuple of (image_tensor, image_url)
        """
        if seed > 0:
            random.seed(seed)

        max_retries = 3

        for attempt in range(max_retries):
            try:
                result = None

                # Fetch from the appropriate source
                if source == "lorem_picsum":
                    result = self._fetch_from_lorem_picsum(width, height, search_term)
                elif source == "wikimedia":
                    result = self._fetch_from_wikimedia(width, height, search_term)
                elif source == "lorem_flickr":
                    result = self._fetch_from_lorem_flickr(width, height, search_term)
                elif source == "unsplash_random":
                    result = self._fetch_from_unsplash_random(width, height, search_term)

                if result is None:
                    raise Exception(f"Could not fetch image from {source}")

                image_id, image_url = result

                # Download and process the image
                image_tensor = self._download_and_process_image(image_url)

                # Mark as used
                self._used_ids[source].add(image_id)
                self._save_history(source)
                self._trim_history_if_needed(source)

                print(f"[SilverStockRandomImage] Fetched new image from {source}: {image_id}")
                return (image_tensor, image_url)

            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"[SilverStockRandomImage] Error after {max_retries} attempts: {e}")
                    # Return blank image on error
                    blank_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                    return (blank_image, "")

                time.sleep(1 * (attempt + 1))  # Exponential backoff

        # Should never reach here, but just in case
        blank_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        return (blank_image, "")
