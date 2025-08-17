# ComfyUI Silver Nodes

A collection of utility nodes for ComfyUI that enhance workflow capabilities with advanced loaders and processing tools.

## Features

Silver Nodes provides several specialized nodes designed to improve your ComfyUI workflows:

### Loading Nodes
- **Silver URL Image Loader**: Load images directly from URLs with cache busting and duplicate prevention
- **Silver Web Image Loader**: Extract the largest image from a webpage
- **Silver Folder Image Loader**: Load images from folders with advanced sorting and batching
- **Silver Folder Video Loader**: Load and process video frames with audio extraction
- **Silver File Text Loader**: Load and process text files with encoding detection
- **Silver Folder File Path Loader**: Get file paths from folders with regex filtering

### Processing Nodes
- **Silver String Replacer**: Perform text replacements using pattern matching
- **Silver Lora Model Loader**: Advanced LORA model loading with filtering and cycling

### Media Nodes
- **Silver Flickr Random Image**: Fetch random images from Flickr with search criteria

### Core Features
- **Efficient Resource Management**: Load and process resources with fine-grained control
- **Batch Processing**: Handle multiple images or models with configurable batch sizes
- **Sequential Processing**: Automatically increment through collections with various strategies
- **Sorting Options**: Organize resources by different criteria
- **Error Handling**: Graceful handling of missing or invalid resources

## Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/yourusername/comfyui-silver-nodes.git
   ```
   
2. Restart ComfyUI

## Nodes

### Silver Lora Model Loader

A specialized LoRA model loader that allows cycling through models with various selection strategies.

#### Features

- **Regex Filtering**: Filter LoRA models by name using regular expressions
- **Sequential Selection**: Increment, decrement, or randomize through filtered models
- **Repeat Control**: Apply the same LoRA multiple times before moving to the next

#### Parameters

- `regex_filter`: Regular expression to filter available LoRA models
- `lora_name`: Currently selected LoRA model
- `action`: Selection strategy (fixed, increment, decrement, randomize)
- `repeat_count`: Number of times to use the same LoRA before changing
- `current_repeat`: Current repeat iteration

### Silver Folder Image Loader

Loads images from a specified folder with advanced sorting and selection options.

#### Features

- **Batch Loading**: Load multiple images at once
- **Flexible Sorting**: Sort by name, creation date, modification date, or file size
- **Sort Direction**: Choose ascending or descending order
- **Selection Strategies**: Fixed position, increment, decrement, or wrap-around

#### Parameters

- `folder_path`: Path to the folder containing images
- `batch_size`: Number of images to load at once
- `current_index`: Starting position in the sorted file list
- `action`: Selection strategy (fixed, increment, decrement, increment_wrap, reset)
- `sort_by`: Sorting criterion (name, created, modified, size)
- `sort_order`: Sort direction (ascending, descending)

### Silver File Text Loader

Loads text from files with options for splitting and sequential selection.

#### Features

- **Text Splitting**: Split by line or paragraph
- **Sequential Access**: Increment, decrement, or randomize through text segments

#### Parameters

- `file_path`: Path to the text file
- `split_mode`: How to split the text (by line, by paragraph)
- `current_index`: Current position in the split text
- `action`: Selection strategy (fixed, increment, decrement, randomize)


## Usage Examples

### Cycling Through LoRA Models

1. Add the Silver Lora Model Loader to your workflow
2. Set a regex filter (e.g., "style.*" to match all style LoRAs)
3. Set action to "increment" to cycle through matching models
4. Connect the output to your workflow

### Processing a Folder of Images

1. Add the Silver Folder Image Loader to your workflow
2. Set the folder path containing your images
3. Configure batch size and sorting options
4. Set action to "increment" to process images sequentially
5. Connect the image output to your processing nodes

### Using Text Prompts from a File

1. Add the Silver File Text Loader to your workflow
2. Set the path to your text file containing prompts
3. Choose split mode based on how your prompts are formatted
4. Set action to "increment" to use a different prompt each run
5. Connect the text output to your prompt input

### Memory-Efficient Batch Processing

1. Add the Silver Batch VAE Decoder after your VAE-using nodes
2. Set an appropriate batch size based on your VRAM
3. Connect to downstream image processing nodes

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.