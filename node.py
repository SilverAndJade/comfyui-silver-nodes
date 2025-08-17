import io
import os
import sys
import re
import random
import time
from datetime import datetime
target_dir = sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import folder_paths
import torch
import numpy as np
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from PIL import Image, ImageOps
from server import PromptServer
import requests
import hashlib
from typing import Any, List, Tuple
import comfy.model_management

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
any_return = AnyType("*")

class SilverRandomFromList:
    @classmethod
    def INPUT_TYPES(cls):
        cls.LORA_NAMES = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "input_list": (any_return),
            }
        }
    
    RETURN_TYPES = (any_return, any_return)
    RETURN_NAMES = ("random_item", "first_item")
    FUNCTION = "get_random_from_list"
    CATEGORY = "silver"

    def get_random_from_list(self, input_list: List[Any]) -> Tuple[Any, Any]:
        """
        Returns a random item from the input list and the first item.
        
        Args:
            input_list (List[Any]): The input list to select from
            
        Returns:
            Tuple[Any, Any]: A tuple containing (random_item, first_item)
        """
        if not input_list:
            return (None, None)
            
        first_item = input_list[0] if input_list else None
        random_item = random.choice(input_list)
        
        return (random_item, first_item)


class SilverLoraModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        cls.LORA_NAMES = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "regex_filter": ("STRING", {"default": ".*"}),
                "lora_name": (cls.LORA_NAMES, ),
                "action": (["fixed", "increment", "decrement", "randomize"], {"default": "increment"}),
                "repeat_count": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "current_repeat": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "fetch_lora_info": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = (any_return, "STRING", "STRING", any_return, any_return)
    RETURN_NAMES = ("COMBO", "STRING", "lora_tags", "example_prompts", "example_images")
    FUNCTION = "load_lora_model"
    CATEGORY = "silver"

    def get_lora_info(self, lora_name):
        try:
            import hashlib
            import requests
            import os
            
            # Get the lora file path
            lora_path = folder_paths.get_full_path("loras", lora_name)
            print(lora_path)
            if not os.path.exists(lora_path):
                return "", [""], [""]
                
            # Calculate hash
            sha256_hash = hashlib.sha256()
            with open(lora_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            hash_value = sha256_hash.hexdigest()
            
            # Query CivitAI
            api_url = f"https://civitai.com/api/v1/model-versions/by-hash/{hash_value}"
            response = requests.get(api_url)
            
            if response.status_code != 200:
                return "", [""], [""]
                
            data = response.json()
            
            # Extract tags
            tags = data.get("trainedWords", [])
            tags_str = ", ".join(tags) if tags else ""
            
            # Extract example prompts
            example_prompts = []
            for image in data.get("images", []):
                if image["hasMeta"]:
                    if "meta" in image and "prompt" in image["meta"]:
                        example_prompts.append(image["meta"]["prompt"])
            
            # Extract example image URLs
            example_images = [img["url"] for img in data.get("images", []) if "url" in img and img["type"] != "video"]
            
            if len(example_prompts) == 0:
                example_prompts.append("")
            if len(example_images) == 0:
                example_images.append("")
            return tags_str, example_prompts, example_images
            
        except Exception as e:
            print(f"Error getting Lora info: {str(e)}")
            return "", [""], [""]

    def load_lora_model(self, regex_filter=".*", lora_name="None", action="fixed", repeat_count=1, current_repeat=0, fetch_lora_info=False):
        lora_names = self.LORA_NAMES
        current_model_index = next((i for i, name in enumerate(lora_names) if name == lora_name), 0)
        regex = re.compile(regex_filter, re.IGNORECASE)
        filtered_loras = [i for i, name in enumerate(lora_names) if regex.search(name)]
        if not filtered_loras:
            return {"ui": {"next_index": [current_model_index], "next_repeat": [current_repeat]},
                    "result": (lora_name, lora_name)}
        try:
            filtered_idx_pos = filtered_loras.index(current_model_index)
        except ValueError:
            filtered_idx_pos = 0
        next_repeat = current_repeat + 1
        next_index = current_model_index
        if next_repeat > repeat_count:
            next_repeat = 1
            max_filtered_pos = len(filtered_loras) - 1
            if action == "increment":
                filtered_idx_pos = (filtered_idx_pos + 1) if filtered_idx_pos < max_filtered_pos else 0
            elif action == "decrement":
                filtered_idx_pos = (filtered_idx_pos - 1) if filtered_idx_pos > 0 else max_filtered_pos
            elif action == "randomize":
                filtered_idx_pos = random.randint(0, max_filtered_pos)
            next_index = filtered_loras[filtered_idx_pos]
        next_model = lora_names[next_index]
        
        lora_tags = ""
        example_prompts = []
        example_images = []
        
        print(next_model)
        if fetch_lora_info:
            lora_tags, example_prompts, example_images = self.get_lora_info(next_model)
        

        return {"ui": {"next_model": [next_model], "next_repeat": [next_repeat]},
                "result": (next_model, next_model, lora_tags, example_prompts, example_images)}

class SilverFolderImageLoader:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
                "batch_size": ("INT", {
                    "default": 1,  
                    "min": 1, 
                    "max": 64, 
                    "step": 1
                }),
                "current_index": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 1000000,
                    "step": 1
                }),
                "action": (["fixed", "increment", "decrement", "increment_wrap", "reset"], ),
                "sort_by": (["name", "created", "modified", "size"], ),
                "sort_order": (["ascending", "descending"], {"default": "ascending"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("images", "filenames", "current_index")
    FUNCTION = "load_images_from_folder"
    CATEGORY = "image"
    
    # Tell ComfyUI this is a stateful node that should always execute
    OUTPUT_NODE = True
    FORCE_ALWAYS_EXECUTE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")  # Always execute

    def __init__(self):
        # Get server instance for state management
        self.server = PromptServer.instance
        if not hasattr(self.server, 'folder_image_indices'):
            self.server.folder_image_indices = {}
        
        # Instance-level storage for current folder state
        self.folder_path = ""
        self.image_files = []
        self.total_images = 0

    def _sort_image_files(self, folder_path, files, sort_by, sort_order="ascending"):
        """Sort image files based on the specified criterion and order."""
        import re
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

        if not files:
            return files

        reverse = (sort_order == "descending")

        if sort_by == "name":
            sorted_files = sorted(files, key=natural_sort_key, reverse=reverse)
        elif sort_by == "created":
            def get_ctime_safe(f):
                try:
                    return os.path.getctime(os.path.join(folder_path, f))
                except Exception:
                    return 0
            sorted_files = sorted(files, key=get_ctime_safe, reverse=reverse)
        elif sort_by == "modified":
            def get_mtime_safe(f):
                try:
                    return os.path.getmtime(os.path.join(folder_path, f))
                except Exception:
                    return 0
            sorted_files = sorted(files, key=get_mtime_safe, reverse=reverse)
        elif sort_by == "size":
            def get_size_safe(f):
                try:
                    return os.path.getsize(os.path.join(folder_path, f))
                except Exception:
                    return 0
            sorted_files = sorted(files, key=get_size_safe, reverse=reverse)
        else:
            sorted_files = sorted(files, key=natural_sort_key, reverse=reverse)

        return sorted_files

    def _get_image_files(self, folder_path, sort_by="name", sort_order="ascending"):
        """Get sorted list of image files in the folder. Always refreshes and sorts."""
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
        try:
            files = [
                f for f in os.listdir(folder_path)
                if os.path.splitext(f)[1].lower() in supported_extensions
            ]
        except Exception as e:
            print(f"[FolderImageLoader] Error listing files in {folder_path}: {e}")
            files = []
        sorted_files = self._sort_image_files(folder_path, files, sort_by, sort_order)
        return sorted_files

    def _process_image(self, img):
        """Process a PIL image into the format expected by ComfyUI."""
        image = ImageOps.exif_transpose(img)
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return image

    def load_images_from_folder(self, folder_path: str, batch_size: int = 1, 
                              current_index: int = 0, action: str = 'fixed',
                              sort_by: str = 'name', sort_order: str = 'ascending'):
        """Load images from a specified folder with flexible index management."""
        if not os.path.isdir(folder_path):
            raise ValueError(f"[FolderImageLoader] Invalid folder path: {folder_path}")

        # Always get freshly sorted file list
        image_files = self._get_image_files(folder_path, sort_by, sort_order)
        if not image_files:
            raise ValueError(f"[FolderImageLoader] No supported image files found in {folder_path}")

        # Ensure index is within bounds after sorting
        current_idx = max(0, min(current_index, len(image_files) - 1))

        # Load images
        loaded_images = []
        filenames = []
        for i in range(batch_size):
            idx = (current_idx + i) % len(image_files)
            image_path = os.path.join(folder_path, image_files[idx])
            try:
                with Image.open(image_path) as img:
                    loaded_images.append(self._process_image(img))
                filenames.append(image_files[idx])
            except Exception as e:
                print(f"[FolderImageLoader] Error loading image {image_path}: {e}")
                continue

        if not loaded_images:
            raise ValueError("[FolderImageLoader] No valid images could be loaded from the specified folder")
        
        # Check for consistent image sizes before stacking
        image_shapes = [img.shape for img in loaded_images]
        if len(set(image_shapes)) > 1:
            raise ValueError("[FolderImageLoader] Different size images detected in the batch. Please only load one image at a time, or ensure all images are the same size.")

        images = torch.cat(loaded_images, dim=0)

        # Update index based on action
        if action == "increment":
            next_index = (current_idx + 1) % len(image_files)
        elif action == "decrement":
            next_index = (current_idx - 1) if current_idx > 0 else len(image_files) - 1
        elif action == "increment_wrap":
            next_index = (current_idx + 1) % len(image_files)
        else:  # "fixed" or "reset"
            next_index = current_idx

        self.current_index = current_idx
        self.next_index = next_index

        return {"ui": {"next_index": [next_index]}, "result": (images, filenames, next_index)}

class SilverFolderVideoLoader:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
                "frame_cap": ("INT", {
                    "default": 0,  
                    "min": 0, 
                    "max": 1000, 
                    "step": 1,
                    "display": "number"
                }),
                "frame_rate_force": ("FLOAT", {
                    "default": 0.0,  
                    "min": 0.0, 
                    "max": 60.0, 
                    "step": 1.0,
                    "display": "number"
                }),
                "start_frame": ("INT", {
                    "default": 0,  
                    "min": 0, 
                    "step": 1,
                    "display": "number"
                }),
                "batch_size": ("INT", {
                    "default": 1,  
                    "min": 1, 
                    "max": 64, 
                    "step": 1,
                    "display": "number"
                }),
                "current_index": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 1000000,
                    "step": 1
                }),
                "action": (["fixed", "increment", "decrement", "increment_wrap", "reset"], ),
                "sort_by": (["name", "created", "modified", "size"], ),
                "sort_order": (["ascending", "descending"], {"default": "ascending"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "FLOAT", "AUDIO")
    RETURN_NAMES = ("images", "filenames", "current_index", "frame_rate", "audio")
    FUNCTION = "load_videos_from_folder"
    CATEGORY = "video"
    
    # Tell ComfyUI this is a stateful node that should always execute
    OUTPUT_NODE = True
    FORCE_ALWAYS_EXECUTE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")  # Always execute

    def __init__(self):
        # Get server instance for state management
        self.server = PromptServer.instance
        if not hasattr(self.server, 'folder_image_indices'):
            self.server.folder_image_indices = {}
        
        # Instance-level storage for current folder state
        self.folder_path = ""
        self.image_files = []
        self.total_images = 0

    def _sort_video_files(self, folder_path, files, sort_by, sort_order="ascending"):
        """Sort video files based on the specified criterion and order."""
        import re
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

        if not files:
            return files

        reverse = (sort_order == "descending")

        if sort_by == "name":
            sorted_files = sorted(files, key=natural_sort_key, reverse=reverse)
        elif sort_by == "created":
            def get_ctime_safe(f):
                try:
                    return os.path.getctime(os.path.join(folder_path, f))
                except Exception:
                    return 0
            sorted_files = sorted(files, key=get_ctime_safe, reverse=reverse)
        elif sort_by == "modified":
            def get_mtime_safe(f):
                try:
                    return os.path.getmtime(os.path.join(folder_path, f))
                except Exception:
                    return 0
            sorted_files = sorted(files, key=get_mtime_safe, reverse=reverse)
        elif sort_by == "size":
            def get_size_safe(f):
                try:
                    return os.path.getsize(os.path.join(folder_path, f))
                except Exception:
                    return 0
            sorted_files = sorted(files, key=get_size_safe, reverse=reverse)
        else:
            sorted_files = sorted(files, key=natural_sort_key, reverse=reverse)

        return sorted_files

    def _get_video_files(self, folder_path, sort_by="name", sort_order="ascending"):
        """Get sorted list of video files in the folder. Always refreshes and sorts."""
        supported_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv']
        try:
            files = [
                f for f in os.listdir(folder_path)
                if os.path.splitext(f)[1].lower() in supported_extensions
            ]
        except Exception as e:
            print(f"[FolderVideoLoader] Error listing files in {folder_path}: {e}")
            files = []
        sorted_files = self._sort_video_files(folder_path, files, sort_by, sort_order)
        return sorted_files

    def _process_image(self, img):
        """Process a PIL image into the format expected by ComfyUI."""
        image = ImageOps.exif_transpose(img)
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return image

    def _extract_frames(self, video_path, frame_cap=0, frame_rate=0.0, start_frame=0):
        """Extract frames and audio from video file with optional frame capping and frame rate control."""
        try:
            import cv2
            from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_audioclips
            import numpy as np
            
            # First, extract audio using moviepy
            audio = None
            try:
                with VideoFileClip(video_path) as video_clip:
                    if video_clip.audio is not None:
                        # Create a copy of the audio to avoid resource cleanup issues
                        audio = video_clip.audio.copy()
            except Exception as e:
                print(f"[FolderVideoLoader] Warning: Could not extract audio from {video_path}: {e}")
            
            # Then extract frames using OpenCV
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame step based on desired frame rate
            frame_step = 1
            if frame_rate > 0 and fps > frame_rate:
                frame_step = int(round(fps / frame_rate))
            
            # Calculate actual frame cap (0 means no cap)
            if frame_cap > 0:
                max_frames = min(frame_cap, total_frames - start_frame)
            else:
                max_frames = total_frames - start_frame
            
            # Set start position
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frames = []
            frame_count = 0
            
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % frame_step == 0:
                    # Convert BGR to RGB and process
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    frames.append(self._process_image(pil_img))
                
                frame_count += 1
                
                # Skip frames if needed
                if frame_step > 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + frame_count * frame_step)
            
            cap.release()
            
            if not frames:
                raise ValueError(f"No frames could be extracted from {video_path}")
                
            return torch.cat(frames, dim=0), fps, audio  # Return frames, fps, and audio
            
        except Exception as e:
            print(f"[FolderVideoLoader] Error extracting frames from {video_path}: {e}")
            raise

    def load_videos_from_folder(self, folder_path: str, frame_cap: int = 0, frame_rate_force: float = 0.0, 
                              start_frame: int = 0, batch_size: int = 1, current_index: int = 0, 
                              action: str = 'fixed', sort_by: str = 'name', sort_order: str = 'ascending'):
        """Load video frames and audio from a specified folder with flexible index management."""
        if not os.path.isdir(folder_path):
            raise ValueError(f"[FolderVideoLoader] Invalid folder path: {folder_path}")

        # Always get freshly sorted file list
        video_files = self._get_video_files(folder_path, sort_by, sort_order)
        if not video_files:
            raise ValueError(f"[FolderVideoLoader] No supported video files found in {folder_path}")

        # Ensure index is within bounds after sorting
        current_idx = max(0, min(current_index, len(video_files) - 1))

        # Load video frames and audio
        all_frames = []
        all_audio = []
        filenames = []
        
        for i in range(batch_size):
            idx = (current_idx + i) % len(video_files)
            video_path = os.path.join(folder_path, video_files[idx])
            try:
                frames, fps, audio = self._extract_frames(video_path, frame_cap, frame_rate_force, start_frame)
                all_frames.append(frames)
                if audio is not None:
                    all_audio.append(audio)
                filenames.append(video_path)
            except Exception as e:
                print(f"[FolderVideoLoader] Error processing video {video_path}: {e}")
                continue

        if not all_frames:
            raise ValueError("[FolderVideoLoader] No valid videos could be processed from the specified folder")
        
        # Stack all frames from all videos
        images = torch.cat(all_frames, dim=0)
        
        # Combine audio if available
        combined_audio = None
        if all_audio:
            from moviepy.audio.AudioClip import concatenate_audioclips
            try:
                # Filter out None audio clips and ensure they have duration > 0
                valid_audio_clips = [clip for clip in all_audio if clip is not None and clip.duration > 0]
                if valid_audio_clips:
                    if len(valid_audio_clips) == 1:
                        combined_audio = valid_audio_clips[0].copy()
                    else:
                        combined_audio = concatenate_audioclips(valid_audio_clips)
            except Exception as e:
                print(f"[FolderVideoLoader] Error combining audio: {e}")
                combined_audio = None

        # Update index based on action
        if action == "increment":
            next_index = (current_idx + 1) % len(video_files)
        elif action == "decrement":
            next_index = (current_idx - 1) if current_idx > 0 else len(video_files) - 1
        elif action == "increment_wrap":
            next_index = (current_idx + 1) % len(video_files)
        else:  # "fixed" or "reset"
            next_index = current_idx

        self.current_index = current_idx
        self.next_index = next_index

        return {
            "ui": {"next_index": [next_index]}, 
            "result": (images, ", ".join(filenames), next_index, fps, combined_audio)
        }

class SilverFolderFilePathLoader:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
                "regex_filter": ("STRING", {"default": ".*", "multiline": False}),
                "batch_size": ("INT", {
                    "default": 1,  
                    "min": 1, 
                    "max": 64, 
                    "step": 1
                }),
                "current_index": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 1000000,
                    "step": 1
                }),
                "action": (["fixed", "increment", "decrement", "increment_wrap", "reset"], ),
                "sort_by": (["name", "created", "modified", "size"], ),
                "sort_order": (["ascending", "descending"], {"default": "ascending"}),
                "wait_for_new": (["enable", "disable"], {"default": "disable"}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("file_paths", "current_index")
    FUNCTION = "load_files_from_folder"
    CATEGORY = "image"
    
    # Tell ComfyUI this is a stateful node that should always execute
    OUTPUT_NODE = True
    FORCE_ALWAYS_EXECUTE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")  # Always execute

    def __init__(self):
        # Get server instance for state management
        self.server = PromptServer.instance
        if not hasattr(self.server, 'folder_image_indices'):
            self.server.folder_image_indices = {}
        if not hasattr(self.server, 'folder_monitor_state'):
            self.server.folder_monitor_state = {}
        
        # Instance-level storage for current folder state
        self.folder_path = ""
        self.total_images = 0
        self.last_seen_files = set()

    def _sort_files(self, folder_path, files, sort_by, sort_order="ascending"):
        """Sort files based on the specified criterion and order."""
        import re
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

        if not files:
            return files

        reverse = (sort_order == "descending")

        if sort_by == "name":
            sorted_files = sorted(files, key=natural_sort_key, reverse=reverse)
        elif sort_by == "created":
            def get_ctime_safe(f):
                try:
                    return os.path.getctime(os.path.join(folder_path, f))
                except Exception:
                    return 0
            sorted_files = sorted(files, key=get_ctime_safe, reverse=reverse)
        elif sort_by == "modified":
            def get_mtime_safe(f):
                try:
                    return os.path.getmtime(os.path.join(folder_path, f))
                except Exception:
                    return 0
            sorted_files = sorted(files, key=get_mtime_safe, reverse=reverse)
        elif sort_by == "size":
            def get_size_safe(f):
                try:
                    return os.path.getsize(os.path.join(folder_path, f))
                except Exception:
                    return 0
            sorted_files = sorted(files, key=get_size_safe, reverse=reverse)
        else:
            sorted_files = sorted(files, key=natural_sort_key, reverse=reverse)

        return sorted_files

    def _get_files(self, folder_path, sort_by="name", sort_order="ascending", regex_filter=".*"):
        """Get sorted list of files in the folder with regex filtering."""
        if not os.path.isdir(folder_path):
            return []
            
        # Get all files in the directory
        files = []
        for f in os.listdir(folder_path):
            file_path = os.path.join(folder_path, f)
            if os.path.isfile(file_path):
                files.append(f)
        
        # Filter by regex if specified
        if regex_filter and regex_filter != "*":
            try:
                regex = re.compile(regex_filter, re.IGNORECASE)
                files = [f for f in files if regex.search(f)]
            except re.error:
                print(f"[FolderFilePathLoader] Invalid regex pattern: {regex_filter}")
        
        # Sort files
        return self._sort_files(folder_path, files, sort_by, sort_order)

    def _get_current_files(self, folder_path, regex_filter):
        """Get current set of files matching the filter."""
        if not os.path.isdir(folder_path):
            return set()
        
        try:
            files = set()
            for f in os.listdir(folder_path):
                file_path = os.path.join(folder_path, f)
                if os.path.isfile(file_path):
                    if not regex_filter or regex_filter == ".*" or re.search(regex_filter, f, re.IGNORECASE):
                        files.add(f)
            return files
        except Exception as e:
            print(f"[FolderFilePathLoader] Error listing files: {e}")
            return set()

    def load_files_from_folder(self, folder_path: str, regex_filter: str = ".*", batch_size: int = 1, 
                             current_index: int = 0, action: str = 'fixed',
                             sort_by: str = 'name', sort_order: str = 'ascending',
                             wait_for_new: str = 'disable'):
        """
        Load files from a specified folder with flexible index management and regex filtering.
        
        Args:
            wait_for_new: If 'enable', will wait for new files to appear in the folder
        """
        if not os.path.isdir(folder_path):
            raise ValueError(f"[FolderFilePathLoader] Invalid folder path: {folder_path}")

        # Get current files
        current_files = self._get_current_files(folder_path, regex_filter)
        
        # If waiting for new files and we've seen these files before
        if wait_for_new == 'enable':
            # Initialize last_seen_files if not already set
            if not hasattr(self, 'last_seen_files'):
                self.last_seen_files = set()
                
            # If we have last_seen_files and no new files, wait for changes
            if self.last_seen_files and not (current_files - self.last_seen_files):
                print(f"[FolderFilePathLoader] Waiting for new files in {folder_path}...")
                start_time = time.time()
                poll_interval = 1.0  # seconds
                max_wait_time = 300.0  # 5 minutes max wait
                
                while time.time() - start_time < max_wait_time:
                    # Check for new files
                    current_files = self._get_current_files(folder_path, regex_filter)
                    new_files = current_files - self.last_seen_files
                    if new_files:
                        print(f"[FolderFilePathLoader] Detected {len(new_files)} new file(s)")
                        break
                        
                    # Check for cancellation
                    if hasattr(self, 'cancelled') and self.cancelled:
                        raise Exception("Operation cancelled by user")
                        
                    # Wait before polling again
                    time.sleep(min(poll_interval, 0.1))  # Don't sleep too long to maintain responsiveness
                else:
                    raise TimeoutError(f"No new files detected in {folder_path} within {max_wait_time} seconds")
        
        # Update last seen files for next time
        self.last_seen_files = current_files
        
        # Get sorted file list with regex filtering
        self.files = self._get_files(folder_path, sort_by, sort_order, regex_filter)
        self.total_files = len(self.files)
        
        if not self.files:
            raise ValueError(f"[FolderFilePathLoader] No supported files found in {folder_path}")

        # Ensure index is within bounds after sorting
        current_idx = max(0, min(current_index, len(self.files) - 1))

        # Get file paths
        file_paths = []
        for i in range(batch_size):
            idx = (current_idx + i) % len(self.files)
            file_path = os.path.join(folder_path, self.files[idx])
            try:
                file_paths.append(file_path)
            except Exception as e:
                print(f"[FolderFilePathLoader] Error getting file path for {file_path}: {e}")
                continue

        # Update index based on action
        if action == "increment":
            next_index = (current_idx + 1) % len(self.files)
        elif action == "decrement":
            next_index = (current_idx - 1) if current_idx > 0 else len(self.files) - 1
        elif action == "increment_wrap":
            next_index = (current_idx + 1) % len(self.files)
        else:  # "fixed" or "reset"
            next_index = current_idx

        self.current_index = current_idx
        self.next_index = next_index

        return {"ui": {"next_index": [next_index]}, "result": (file_paths, next_index)}

class SilverFileTextLoader:
    """
    Loads text from files with support for multiple encodings and flexible text splitting.
    Handles various text file formats and provides robust error handling for different encodings.
    """
    
    # Common text file extensions
    TEXT_EXTENSIONS = {
        '.txt', '.md', '.csv', '.log', '.srt', '.json', '.xml', '.html', 
        '.yaml', '.yml', '.ini', '.cfg', '.conf', '.py', '.js', '.css', '.tsv'
    }
    
    # Common text encodings to try (in order of likelihood)
    ENCODINGS = [
        'utf-8-sig',  # UTF-8 with BOM
        'utf-8',      # Standard UTF-8
        'utf-16',     # UTF-16 (common in Windows)
        'cp1252',     # Windows Western European
        'latin-1',    # ISO-8859-1
        'iso-8859-1', # Alternative name for latin-1
        'ascii'       # Fallback to ASCII
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "dynamicPrompts": False,
                    "placeholder": "Path to text file"
                }),
                "split_mode": (["by line", "by paragraph"], {"default": "by line"}),
                "current_index": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 1000000, 
                    "step": 1,
                    "display": "number"
                }),
                "action": (["fixed", "increment", "decrement", "randomize"], {
                    "default": "fixed"
                })
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("text", "index", "count")
    OUTPUT_NODE = True
    FORCE_ALWAYS_EXECUTE = True
    FUNCTION = "load_text_from_file"
    CATEGORY = "silver_nodes/loaders"

    def _try_decodings(self, file_path, encodings=None):
        """Try to read a file with multiple encodings."""
        if encodings is None:
            encodings = self.ENCODINGS
            
        errors = []
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                print(f"[FileTextLoader] Successfully read file with {encoding} encoding: {file_path}")
                return content, encoding
            except UnicodeDecodeError as e:
                errors.append(f"{encoding}: {str(e)}")
            except Exception as e:
                errors.append(f"{encoding}: {str(e)}")
                
        # If we get here, all encodings failed
        error_msg = "\n".join([f"  - {e}" for e in errors])
        raise ValueError(
            f"[FileTextLoader] Failed to decode file with any encoding: {file_path}\n"
            f"Tried encodings:\n{error_msg}"
        )

    def _split_text(self, text, mode):
        """Split text into lines or paragraphs."""
        if not text.strip():
            return []
            
        if mode == "by line":
            # Split by lines, remove empty lines
            return [line for line in text.splitlines() if line.strip()]
        else:
            # Split by paragraphs (multiple newlines with optional whitespace)
            paragraphs = re.split(r'\s*\n\s*\n\s*', text)
            return [p.strip() for p in paragraphs if p.strip()]

    def _warn_unusual_extension(self, file_path):
        """Warn if the file has an unusual extension for a text file."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext and ext not in self.TEXT_EXTENSIONS:
            print(f"[FileTextLoader] Warning: Unusual file extension '{ext}' for text file: {file_path}")

    def load_text_from_file(self, file_path: str, split_mode: str = "by line", 
                          current_index: int = 0, action: str = "fixed"):
        """
        Load text from a file with support for multiple encodings and text splitting modes.
        
        Args:
            file_path: Path to the text file
            split_mode: How to split the text ('by line' or 'by paragraph')
            current_index: Current index in the split items
            action: Action to take on the index ('fixed', 'increment', 'decrement', 'randomize')
            
        Returns:
            Dictionary with text, index, and count of items
        """
        # Input validation
        if not file_path or not file_path.strip():
            raise ValueError("[FileTextLoader] No file path provided")
            
        file_path = file_path.strip()
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[FileTextLoader] File does not exist: {file_path}")
            
        if os.path.isdir(file_path):
            raise IsADirectoryError(f"[FileTextLoader] Path is a directory, not a file: {file_path}")
            
        # Check file size to prevent loading huge files
        try:
            file_size = os.path.getsize(file_path)
            if file_size > 10 * 1024 * 1024:  # 10MB limit
                raise ValueError(f"[FileTextLoader] File is too large ({file_size/1024/1024:.1f}MB). Maximum size is 10MB.")
        except Exception as e:
            print(f"[FileTextLoader] Warning: Could not check file size: {str(e)}")
        
        # Warn about unusual file extensions but continue anyway
        self._warn_unusual_extension(file_path)
        
        # Try to read the file with multiple encodings
        try:
            text, used_encoding = self._try_decodings(file_path)
            print(f"[FileTextLoader] Successfully decoded file with {used_encoding} encoding")
        except Exception as e:
            raise ValueError(f"[FileTextLoader] Failed to read file: {file_path}\nError: {str(e)}")
        
        # Split text according to mode
        try:
            items = self._split_text(text, split_mode)
            if not items:
                raise ValueError(f"No content found after splitting by '{split_mode}'. File may be empty or contain only whitespace.")
        except Exception as e:
            raise ValueError(f"[FileTextLoader] Error processing text in file: {file_path}\nError: {str(e)}")
        
        # Handle index and action
        max_index = len(items) - 1
        if max_index < 0:
            return {"ui": {"current_index": [0]}, "result": ("", 0, 0)}
            
        idx = max(0, min(current_index, max_index))
        
        if action == "increment":
            idx = (idx + 1) % (max_index + 1)
        elif action == "decrement":
            idx = (idx - 1) % (max_index + 1)
        elif action == "randomize":
            idx = random.randint(0, max_index)
        # else fixed: do nothing

        selected = items[idx]
        print(f"[FileTextLoader] Loaded item {idx + 1}/{len(items)} from '{file_path}' (mode: {split_mode}, action: {action})")
        
        return {
            "ui": {
                "next_index": [idx],
                "total_items": [len(items)]
            }, 
            "result": (selected, idx, len(items))
        }

class SilverFlickrRandomImage:
    """
    A ComfyUI node that fetches a random image from Flickr based on search criteria.
    Uses the public Flickr API with a predefined API key.
    """
    
    # Flickr API configuration
    FLICKR_API_KEY = "d61091b4772d0e40c3743b6a5fc54084"
    FLICKR_API_BASE = "https://www.flickr.com/services/rest/"
    _history_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'history', 'flickr_history.txt')
    _loaded_hashes = set()
    
    @classmethod
    def _load_history(cls):
        """Load the history of loaded image hashes from file."""
        try:
            os.makedirs(os.path.dirname(cls._history_file), exist_ok=True)
            if os.path.exists(cls._history_file):
                with open(cls._history_file, 'r') as f:
                    cls._loaded_hashes = set(line.strip() for line in f if line.strip())
        except Exception as e:
            print(f"Warning: Failed to load Flickr history: {e}")

    @classmethod
    def _save_to_history(cls, content_hash):
        """Save a content hash to the history file with rolling history (max 1000 entries)."""
        try:
            # Ensure history directory exists
            os.makedirs(os.path.dirname(cls._history_file), exist_ok=True)
            
            # Read existing hashes if file exists
            hashes = set()
            if os.path.exists(cls._history_file):
                with open(cls._history_file, 'r') as f:
                    hashes = set(line.strip() for line in f if line.strip())
            
            # Add the new hash
            hashes.add(content_hash)
            
            # If we have more than 1000 hashes, keep only the 900 most recent
            if len(hashes) > 1000:
                # Convert to list and keep the most recent 900
                hashes = list(hashes)[-900:]
                
            # Write back to file
            with open(cls._history_file, 'w') as f:
                for h in hashes:
                    f.write(f"{h}\n")
                    
            # Update the in-memory set
            cls._loaded_hashes = set(hashes)
            
        except Exception as e:
            print(f"Warning: Failed to save to Flickr history: {e}")
            
    def __init__(self):
        self._load_history()
        
    def _get_content_hash(self, photo_id):
        """Generate a unique hash for the photo to prevent duplicates."""
        return hashlib.md5(photo_id.encode('utf-8')).hexdigest()
    
    @classmethod
    def INPUT_TYPES(cls):
        # Set default date range (2001-01-01 to today)
        now = datetime.now()
        default_min_date = "2001-01-01"
        default_max_date = now.strftime("%Y-%m-%d")
        
        # Available styles from Flickr - include 'any' as first option
        styles = [
            "any",  # First option for "any style"
            "blackandwhite", "depthoffield", "minimalism", "pattern",
            "minimal", "abstract", "macro", "bokeh", "moody", "vibrant",
            "pastel", "monochrome", "hdr", "longexposure", "silhouette"
        ]
        
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": False, "dynamicPrompts": False}),
                "min_upload_date": ("STRING", {"default": default_min_date, "multiline": False}),
                "max_upload_date": ("STRING", {"default": default_max_date, "multiline": False}),
                "orientation": (["any", "landscape", "portrait", "square", "panorama"], {"default": "any"}),
                "styles": (styles, {"default": "any", "multiselect": True}),
                "safe_search": (["Safe", "Moderate", "Restricted"], {"default": "Safe"}),
                "per_page": ("INT", {"default": 100, "min": 1, "max": 500, "display": "number"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32-1, "display": "number"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "image_url")
    FUNCTION = "fetch_image"
    CATEGORY = "image/loading"
    
    def _parse_date(self, date_str):
        """Parse date string from YYYY-MM-DD to timestamp."""
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return int(dt.timestamp())
        except (ValueError, TypeError):
            # Fallback to current date if parsing fails
            return int(time.time())
    
    def _process_styles(self, styles):
        """Process and validate styles parameter."""
        # If 'any' is selected or no styles are specified, include all available styles
        if not styles or "any" in styles:
            return ""
        return ",".join(styles)
    
    def fetch_image(self, text, min_upload_date, max_upload_date, orientation, styles, safe_search, per_page, seed=0):
        """Fetch a random image from Flickr with retry logic and duplicate prevention.
        
        Args:
            seed: Random seed for reproducible results. If 0, uses system time.
        """
        if seed > 0:
            random.seed(seed)
            
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                # Convert safe search text to Flickr's numeric format
                safe_search_map = {
                    "Safe": "1",
                    "Moderate": "2",
                    "Restricted": "3"
                }
                safe_search_value = safe_search_map.get(safe_search, "1")  # Default to Safe
                
                # Parse dates
                min_date = self._parse_date(min_upload_date)
                max_date = self._parse_date(max_upload_date)
                
                # Ensure min_date is not greater than max_date
                if min_date > max_date:
                    min_date, max_date = max_date, min_date
                
                # Process styles
                styles_value = self._process_styles(styles)
                
                # Build search parameters - request multiple sizes to ensure we get at least one
                params = {
                    "method": "flickr.photos.search",
                    "api_key": self.FLICKR_API_KEY,
                    "text": text,
                    "min_upload_date": str(min_date),
                    "max_upload_date": str(max_date),
                    "safe_search": safe_search_value,
                    "media": "photos",
                    "per_page": min(per_page, 100),  # Reduced per_page since we'll try multiple pages
                    "format": "json",
                    "nojsoncallback": 1,
                    "extras": "url_b,url_c,url_z,url_m,url_n,url_o"  # Multiple sizes for fallback, including original
                }
                
                # Handle orientation - if 'any', include all orientations
                if orientation == "any":
                    params["orientation"] = "landscape,portrait,square,panorama"
                else:
                    params["orientation"] = orientation
                    
                # Add styles parameter if we have any
                if styles_value:
                    params["styles"] = styles_value
                    
                # Add empty license parameter as per API requirements
                params["license"] = ""
                
                # First, get total number of pages
                search_response = self._flickr_api_call(params)
                if not search_response or "photos" not in search_response:
                    raise Exception("Failed to fetch images from Flickr")
                
                total_pages = min(int(search_response["photos"]["pages"]), 10)  # Limit to 10 pages max
                if total_pages == 0:
                    raise Exception("No images found matching the criteria")
                
                # Try up to 3 random pages to find a new image
                for _ in range(3):
                    # Get a random page
                    page = random.randint(1, total_pages)
                    params["page"] = page
                    
                    page_response = self._flickr_api_call(params)
                    if not page_response or "photos" not in page_response:
                        continue
                        
                    photos = page_response["photos"]["photo"]
                    if not photos:
                        continue
                    
                    # Shuffle photos to get random order
                    random.shuffle(photos)
                    
                    # Find a photo we haven't seen before
                    for photo in photos:
                        photo_hash = self._get_content_hash(photo["id"])
                        if photo_hash not in self._loaded_hashes:
                            # Try different sizes in order of preference (largest to smallest, with original as final fallback)
                            for size in ['url_b', 'url_c', 'url_z', 'url_m', 'url_n', 'url_o']:
                                image_url = photo.get(size)
                                if image_url:
                                    break  # Found a valid URL
                            
                            if not image_url:
                                print(f"No suitable image URL found for photo {photo.get('id')}")
                                continue
                                
                            # Download and process the image
                            image = self._download_and_process_image(image_url)
                            
                            # Save to history before returning
                            self._save_to_history(photo_hash)
                            return (image, image_url)
                
                # If we get here, we've tried multiple pages but couldn't find a new image
                # Clear history if we've seen too many images
                if len(self._loaded_hashes) > 1000:
                    self._loaded_hashes.clear()
                    if os.path.exists(self._history_file):
                        os.remove(self._history_file)
                        
                raise Exception("No new images found after multiple attempts")
                
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    print(f"Error in FlickrRandomImage after {max_retries} attempts: {str(e)}")
                    # Return a blank image on error
                    blank_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                    return (blank_image, "")
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
    
    def _flickr_api_call(self, params):
        try:
            # Log the full URL with query parameters for debugging
            from urllib.parse import urlencode
            base_url = "https://www.flickr.com/services/rest/"
            full_url = f"{base_url}?{urlencode(params, doseq=True)}"
            print(f"[FlickrRandomImage] Making API request to: {full_url}")
            
            # First, get total number of pages
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            print(f"Flickr API request failed: {e}")
            return None
       
    def _download_and_process_image(self, image_url):
        """Download and process the image into a tensor."""
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            # Load image
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            
            # Resize if too large (to prevent memory issues)
            max_size = 2048
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to tensor
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
            return image_tensor
            
        except Exception as e:
            print(f"Error processing image: {e}")
            raise

class SilverWebImageLoader:
    """
    A ComfyUI node that loads the largest image from a webpage URL.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "", "multiline": False, "dynamicPrompts": False}),
                "exclude": ("STRING", {"default": "nav,logo,header,footer,banner,menu,thumbnail,thumb,icon,button,social,share,ad,ads,advert,sponsor,sponsored,widget,related,recommended,popular,trending,static,gif", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32-1, "display": "number"}),
            },
            "optional": {
                "timeout": ("INT", {"default": 10, "min": 1, "max": 60, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    CATEGORY = "image/loading"
    
    @classmethod
    def _get_content_hash(cls, content):
        """Generate a hash of the image content for deduplication."""
        import hashlib
        return hashlib.sha256(content).hexdigest()
    
    def _get_history_file(self):
        """Get the path to the image hash history file."""
        import os
        history_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'history')
        os.makedirs(history_dir, exist_ok=True)
        return os.path.join(history_dir, 'webimageloader_history.txt')

    def _load_history(self):
        """Load the history of loaded image hashes from file."""
        if not hasattr(self, '_loaded_hashes'):
            self._loaded_hashes = set()
            try:
                history_file = self._get_history_file()
                if os.path.exists(history_file):
                    with open(history_file, 'r') as f:
                        self._loaded_hashes = set(line.strip() for line in f if line.strip())
                    print(f"[Debug] Loaded {len(self._loaded_hashes)} image hashes from history")
            except Exception as e:
                print(f"Warning: Could not read image hash history: {e}")
        return self._loaded_hashes
    
    def _save_to_history(self, content_hash):
        """Save an image hash to the history file with rolling history (max 1000 entries)."""
        if not hasattr(self, '_loaded_hashes'):
            self._load_history()
            
        if content_hash in self._loaded_hashes:
            return
            
        try:
            history_file = self._get_history_file()
            
            # Read existing hashes
            hashes = set()
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    hashes = set(line.strip() for line in f if line.strip())
            
            # Add the new hash
            hashes.add(content_hash)
            
            # If we have more than 1000 hashes, keep only the 900 most recent
            if len(hashes) > 1000:
                # Convert to list and keep the most recent 900
                hashes = list(hashes)[-900:]
            
            # Write back to file
            with open(history_file, 'w') as f:
                for h in hashes:
                    f.write(f"{h}\n")
            
            # Update in-memory set
            self._loaded_hashes = hashes
            
        except Exception as e:
            print(f"Warning: Could not update image hash history: {e}")
            
    def __init__(self):
        super().__init__()
        self._load_history()

    def _get_image_size_info(self, img_element, base_url):
        """Extract size information from an img element without downloading the image."""
        size_info = {
            'width': None,
            'height': None,
            'area': 0,
            'url': None
        }
        
        # Get the image URL first
        img_url = img_element.get('src', '')
        if not img_url:
            return None
            
        # Normalize URL
        if img_url.startswith('//'):
            img_url = f"https:{img_url}"  # Assume https for protocol-relative URLs
        elif not img_url.startswith(('http://', 'https://')):
            img_url = urljoin(base_url, img_url)
        
        size_info['url'] = img_url
        
        # 1. Check explicit width/height attributes
        width = img_element.get('width')
        height = img_element.get('height')
        
        # 2. Check inline styles
        if not all([width, height]):
            style = img_element.get('style', '')
            if style:
                width_match = re.search(r'width\s*:\s*(\d+)px', style)
                height_match = re.search(r'height\s*:\s*(\d+)px', style)
                width = width or (width_match.group(1) if width_match else None)
                height = height or (height_match.group(1) if height_match else None)
        
        # 3. Check data attributes (common in lazy-loaded images)
        if not all([width, height]):
            width = img_element.get('data-width') or img_element.get('data-original-width')
            height = img_element.get('data-height') or img_element.get('data-original-height')
        
        # Convert to integers if we have both dimensions
        try:
            if width and height:
                width = int(str(width).replace('px', '').strip())
                height = int(str(height).replace('px', '').strip())
                size_info.update({
                    'width': width,
                    'height': height,
                    'area': width * height
                })
        except (ValueError, TypeError):
            pass
            
        return size_info

    def get_largest_image(self, url, exclude, timeout=10, seed=0, always_unique=False):
        """Fetch the largest image from the given URL with retry logic.
        
        Args:
            url: The URL of the webpage to fetch images from
            exclude: Comma-separated keywords to exclude from image URLs
            timeout: Request timeout in seconds
            seed: Random seed for consistent selection (0 for random)
            always_unique: Not used, kept for compatibility
            
        Returns:
            PIL.Image: The downloaded image
        """
        import random
        import time
        from urllib.parse import urlparse, urljoin
        
        # Set up session
        session = requests.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        }
        
        # Set random seed if provided
        if seed > 0:
            random.seed(seed)
        
        # Get exclude keywords
        exclude_keywords = {kw.strip().lower() for kw in exclude.split(',') if kw.strip()}
        exclude_keywords.update({'sprite', 'thumbs'})
        
        max_retries = 5
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Load history if not already loaded
                self._load_history()
                
                # Fetch the webpage
                response = session.get(url, headers=headers, timeout=timeout)
                response.raise_for_status()
                
                # Parse HTML and find all img elements
                soup = BeautifulSoup(response.text, 'html.parser')
                img_elements = soup.find_all('img', src=True)
                
                if not img_elements:
                    raise ValueError("No image elements found on the page")
                
                # Extract and process image information
                base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
                image_infos = []
                
                for img in img_elements:
                    # Get size info and URL
                    size_info = self._get_image_size_info(img, base_url)
                    if not size_info or not size_info['url']:
                        continue
                        
                    # Skip excluded URLs
                    img_url_lower = size_info['url'].lower()
                    if any(kw in img_url_lower for kw in exclude_keywords):
                        continue
                        
                    image_infos.append(size_info)
                
                if not image_infos:
                    raise ValueError("No valid image URLs found after filtering")
                
                # Sort by area (largest first), then by URL for consistency
                image_infos.sort(key=lambda x: (-x['area'], x['url']))
                
                # Try each image URL until one succeeds, starting with the largest
                for img_info in image_infos:
                    try:
                        img_url = img_info['url']
                        img_response = session.get(img_url, headers=headers, timeout=timeout)
                        img_response.raise_for_status()
                        
                        # Verify it's actually an image
                        if not img_response.headers.get('content-type', '').startswith('image/'):
                            print(f"Skipping non-image URL: {img_url}")
                            continue
                            
                        image_content = img_response.content
                        content_hash = self._get_content_hash(image_content)
                        
                        # Skip if we've already loaded this exact image
                        if content_hash in self._loaded_hashes:
                            print(f"Skipping duplicate image with hash: {content_hash}")
                            continue
                            
                        # Add to history before returning
                        self._save_to_history(content_hash)
                            
                        return Image.open(io.BytesIO(image_content)).convert("RGB")
                        
                    except Exception as e:
                        print(f"Failed to download {img_url}: {e}")
                        continue
                
                raise ValueError("Could not download any of the found images")
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Attempt {attempt + 1} failed, retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                raise Exception(f"Failed after {max_retries} attempts. Last error: {str(e)}")
        
        # If we get here, all attempts failed
        if last_error:
            raise last_error
        raise Exception("Failed to load image")

    def load_image(self, url, exclude="", seed=0, timeout=10):
        """Load an image from a URL with optional random seed for selection.
        
        Args:
            url: The URL to load images from
            seed: Random seed for consistent selection (0 for random)
            timeout: Request timeout in seconds
        """
        # Validate URL
        if not url or not isinstance(url, str) or not url.strip():
            raise ValueError("Please provide a valid URL")
            
        # Get the image (largest or selected by seed)
        image = self.get_largest_image(url.strip(), exclude, timeout, seed)
        
        # Convert to tensor
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]
        
        # Create a mask (all ones for RGB images)
        mask = torch.ones((64, 64), dtype=torch.float32, device="cpu")
        
        return (image_tensor, mask.unsqueeze(0))

class SilverUrlImageLoader:
    """
    A ComfyUI node that loads an image directly from a URL with cache busting.
    Tracks previously loaded images to avoid redundant downloads.
    """
    _loaded_hashes = set()
    _history_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'history', 'urlimageloader_history.txt')
    
    @classmethod
    def _load_history(cls):
        """Load the history of loaded image hashes from file."""
        try:
            os.makedirs(os.path.dirname(cls._history_file), exist_ok=True)
            if os.path.exists(cls._history_file):
                with open(cls._history_file, 'r') as f:
                    cls._loaded_hashes = set(line.strip() for line in f if line.strip())
        except Exception as e:
            print(f"Warning: Failed to load image history: {e}")
    
    @classmethod
    def _save_to_history(cls, content_hash):
        """Save a content hash to the history file with rolling history (max 1000 entries)."""
        try:
            # Ensure history directory exists
            os.makedirs(os.path.dirname(cls._history_file), exist_ok=True)
            
            # Read existing hashes if file exists
            hashes = set()
            if os.path.exists(cls._history_file):
                with open(cls._history_file, 'r') as f:
                    hashes = set(line.strip() for line in f if line.strip())
            
            # Add the new hash
            hashes.add(content_hash)
            
            # If we have more than 1000 hashes, keep only the 900 most recent
            if len(hashes) > 1000:
                # Convert to list and keep the most recent 900
                hashes = list(hashes)[-900:]
                
            # Write back to file
            with open(cls._history_file, 'w') as f:
                for h in hashes:
                    f.write(f"{h}\n")
                    
            # Update the in-memory set
            cls._loaded_hashes = set(hashes)
            
        except Exception as e:
            print(f"Warning: Failed to save to image history: {e}")
    
    def __init__(self):
        super().__init__()
        self._load_history()
    
    @classmethod
    def _get_content_hash(cls, content):
        """Generate a hash of the image content for deduplication."""
        return hashlib.sha256(content).hexdigest()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "", "multiline": False, "dynamicPrompts": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32-1, "display": "number"}),
            },
            "optional": {
                "timeout": ("INT", {"default": 10, "min": 1, "max": 60, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    CATEGORY = "image/loading"
    
    def load_image(self, url, seed=0, timeout=10):
        """Load an image directly from a URL with cache busting.
        
        Args:
            url: The URL of the image to load
            seed: Initial random seed for cache busting (0 = no cache busting)
            timeout: Request timeout in seconds
            
        Returns:
            Tuple of (image_tensor, mask_tensor)
            
        Raises:
            Exception: If image loading fails after all retry attempts
        """
        import requests
        from io import BytesIO
        from PIL import Image, ImageOps
        import torch
        import random
        
        if not url:
            raise comfy.model_management.InterruptProcessingException("Skipping workflow: URL is empty")

        max_retries = 5
        current_seed = seed
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Generate a new seed for each retry after the first attempt
                if attempt > 0:
                    current_seed = random.randint(1, 2**32-1)
                
                # Add cache-busting parameter if seed is provided
                current_url = url
                if current_seed > 0:
                    parsed = urlparse(current_url)
                    if parsed.query:
                        current_url = f"{current_url}&_cb={current_seed}"
                    else:
                        current_url = f"{current_url}?_cb={current_seed}"
                
                # Set up session with cache-busting headers
                session = requests.Session()
                session.headers.update({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    'Pragma': 'no-cache',
                    'Expires': '0'
                })
                
                # Download the image
                response = session.get(current_url, timeout=timeout, stream=True)
                response.raise_for_status()
                
                content = response.content
                content_hash = self._get_content_hash(content)
                
                # Skip if we've already loaded this exact image
                if content_hash in self._loaded_hashes:
                    raise Exception(f"Image with hash {content_hash} has already been loaded")
                
                # Load image
                img = Image.open(BytesIO(content)).convert("RGB")
                
                # Convert to tensor
                img = ImageOps.exif_transpose(img)
                img = np.array(img).astype(np.float32) / 255.0
                img = torch.from_numpy(img)[None,]
                
                # Create a mask of ones (fully visible)
                mask = torch.ones((64, 64), dtype=torch.float32, device="cpu")
                
                # Add hash to loaded hashes and save to history
                self._loaded_hashes.add(content_hash)
                self._save_to_history(content_hash)
                
                return (img, mask)
                
            except Exception as e:
                last_error = e
                if attempt == max_retries - 1:  # Last attempt
                    break
                continue
        
        # If we get here, all attempts failed
        raise comfy.model_management.InterruptProcessingException(
            f"Skipping workflow: Failed to load image from URL after {max_retries} attempts"
        )

class SilverStringReplacer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "replacement_pattern": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "replace_strings"
    CATEGORY = "silver_nodes/text"

    def replace_strings(self, text, replacement_pattern):
        if not replacement_pattern.strip():
            return (text,)
            
        # Create a mapping of search terms to their possible replacements
        replacement_map = {}
        for line in replacement_pattern.split('\n'):
            parts = [p.strip() for p in line.split(':') if p.strip()]
            if len(parts) >= 2:
                replacement_map[parts[0]] = parts[1:]
        
        if not replacement_map:
            return (text,)
            
        # Replace occurrences in the input text
        result = text
        for search_term, replacements in replacement_map.items():
            if search_term in result:
                replacement = random.choice(replacements)
                result = result.replace(search_term, replacement)
                
        return (result,)

NODE_CLASS_MAPPINGS = {
    # ... your existing mappings
    "SilverStringReplacer": SilverStringReplacer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # ... your existing mappings
    "SilverStringReplacer": "Silver String Replacer",
}

NODE_CLASS_MAPPINGS = {
    "SilverStringReplacer": SilverStringReplacer,
    "SilverLoraModelLoader": SilverLoraModelLoader,
    "SilverFolderImageLoader": SilverFolderImageLoader,
    "SilverFolderVideoLoader": SilverFolderVideoLoader,
    "SilverFolderFilePathLoader": SilverFolderFilePathLoader,
    "SilverFileTextLoader": SilverFileTextLoader,
    "SilverFlickrRandomImage": SilverFlickrRandomImage,
    "SilverWebImageLoader": SilverWebImageLoader,
    "SilverUrlImageLoader": SilverUrlImageLoader,
    "SilverRandomFromList": SilverRandomFromList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SilverStringReplacer": "Silver String Replacer",
    "SilverLoraModelLoader": "Silver Lora Model Loader",
    "SilverFolderImageLoader": "Silver Folder Image Loader",
    "SilverFolderVideoLoader": "Silver Folder Video Loader",
    "SilverFolderFilePathLoader": "Silver Folder File Path Loader",
    "SilverFileTextLoader": "Silver File Text Loader",
    "SilverFlickrRandomImage": "Silver Flickr Random Image",
    "SilverWebImageLoader": "Silver Load Largest Web Image",
    "SilverUrlImageLoader": "Silver URL Image Loader",
    "SilverRandomFromList": "Silver Random From List"
}
