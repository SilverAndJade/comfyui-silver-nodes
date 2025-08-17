# Changelog

All notable changes to the comfyui-silver-nodes project will be documented in this file.

## [Unreleased]

### Added
- **Silver URL Image Loader**: New node for loading images directly from URLs
  - Added cache busting to prevent browser caching issues
  - Implemented duplicate detection to avoid loading the same image multiple times
  - Added support for graceful error handling with `InterruptProcessingException`
  - Included retry mechanism for failed downloads

- **Silver Web Image Loader**: New node to extract the largest image from a webpage
  - Parses HTML to find the largest image on a webpage
  - Includes options to exclude specific image types or classes
  - Handles relative URLs and various image formats

- **Enhanced Error Handling**: Improved error handling across all nodes
  - Added `InterruptProcessingException` for graceful workflow interruption
  - Implemented retry mechanisms for network operations
  - Added detailed error messages for troubleshooting

### Changed
- **Silver Folder Video Loader**: 
  - Updated to be compatible with moviepy 2.2.1
  - Added proper resource management with context managers
  - Improved audio extraction reliability
  - Enhanced error handling for corrupt video files

### Fixed
- **Silver Lora Model Loader**:
  - Fixed issue with model caching
  - Improved regex filter performance
  - Fixed edge cases in model cycling logic

- **Silver Folder Image Loader**:
  - Fixed sorting by date modified
  - Improved handling of unsupported image formats
  - Fixed batch processing with empty folders

## [1.0.0] - 2025-08-17

### Added
- **Initial Release**: First public release of Silver Nodes
  - Core set of nodes for image, video, and text processing
  - Support for batch processing and sequential workflows
  - Comprehensive documentation and examples

### Changed
- Updated `load_videos_from_folder` method to handle audio data alongside video frames
- Improved error handling and logging for audio-related operations
- Modified return structure to include audio in the result tuple

### Dependencies
- Added `moviepy>=2.2.1` as a required dependency for audio processing

## How to Revert Changes

To revert the audio extraction changes, you would need to:

1. Revert the changes to the following methods in `node.py`:
   - `_extract_frames`: Remove audio extraction logic and restore original return values
   - `load_videos_from_folder`: Remove audio handling and restore original return structure
   - Revert the `RETURN_TYPES` and `RETURN_NAMES` to their original values

2. Remove `moviepy` from your dependencies if it's not needed elsewhere.

3. Update any nodes that might be consuming the audio output from the `FolderVideoLoader`.

## Versioning

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-07-28
### Added
- Initial release of comfyui-silver-nodes
- Basic video loading functionality with `FolderVideoLoader` class
