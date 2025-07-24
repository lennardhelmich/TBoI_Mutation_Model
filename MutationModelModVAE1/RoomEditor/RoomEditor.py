import os
import time
import json
import math
import numpy as np
import random
import torch
from pathlib import Path
from PIL import Image
from tboi_bitmap import TBoI_Bitmap, EntityType

# Add these imports for the VAE functionality
try:
    from tboi_vae import ConvVAE, load_model, DEVICE
except ImportError:
    print("Warning: tboi_vae module not found. VAE functionality will be limited.")
    DEVICE = "cpu"

# Predefined bitmap: 15x9 room with border of 0s (WALL) and interior of 2s (FREE_SPACE)
CONSTANT_BITMAP = """0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
0,2,2,2,2,2,2,2,2,2,2,2,2,2,0
0,2,2,2,2,2,2,2,2,2,2,2,2,2,0
0,2,2,2,2,2,2,2,2,2,2,2,2,2,0
0,2,2,2,2,2,2,2,2,2,2,2,2,2,0
0,2,2,2,2,2,2,2,2,2,2,2,2,2,0
0,2,2,2,2,2,2,2,2,2,2,2,2,2,0
0,2,2,2,2,2,2,2,2,2,2,2,2,2,0
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"""

class SaveFileMonitor:
    def __init__(self, file_path, on_room_change_callback=None):
        self.file_path = Path(file_path)
        self.last_modified = 0
        self.last_content = ""
        self.last_bitmap = ""  # Store the last bitmap for comparison
        self.on_room_change_callback = on_room_change_callback
        

    def check_file_exists(self):
        """Check if the save file exists"""
        return self.file_path.exists()
    
    def get_file_modified_time(self):
        """Get the last modified time of the file"""
        try:
            return os.path.getmtime(self.file_path)
        except OSError:
            return 0
    
    def read_file_content(self):
        """Read and return the content of the save file"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                return content
        except (FileNotFoundError, PermissionError, UnicodeDecodeError) as e:
            print(f"Error reading file: {e}")
            return None
    
    def parse_content(self, content):
        """Try to parse content as JSON, fallback to raw text"""
        if not content:
            return "Empty file"
        
        try:
            # Try to parse as JSON
            parsed = json.loads(content)
            return json.dumps(parsed, indent=2)
        except json.JSONDecodeError:
            # If not JSON, return as plain text with formatting
            return self.format_plain_text(content)
    
    def format_plain_text(self, content):
        """Format plain text content for better readability"""
        # If it looks like a bitmap string (contains commas and newlines)
        if ',' in content and '\n' in content:
            lines = content.split('\n')
            formatted = "Room Bitmap Data:\n"
            for i, line in enumerate(lines):
                formatted += f"Row {i:2d}: {line}\n"
            return formatted
        else:
            return f"Raw content:\n{content}"
    
    def monitor(self):
        """Main monitoring loop"""
        print(f"Monitoring save file: {self.file_path}")
        print("Waiting for changes... (Press Ctrl+C to stop)")
        print("-" * 50)
        
        while True:
            try:
                if not self.check_file_exists():
                    if self.last_content:  # File was deleted
                        print("Save file deleted!")
                        self.last_content = ""
                        self.last_modified = 0
                    time.sleep(1)
                    continue
                
                current_modified = self.get_file_modified_time()
                
                # Check if file has been modified
                if current_modified > self.last_modified:
                    content = self.read_file_content()
                    
                    if content is not None and content != self.last_content:
                        print(f"\n[{time.strftime('%H:%M:%S')}] Save file changed!")
                        print("-" * 50)
                        
                        formatted_content = self.parse_content(content)
                        print(formatted_content)
                        
                        print("-" * 50)
                        
                        # Call the callback function if provided
                        if self.on_room_change_callback:
                            try:
                                # If it's bitmap data, also provide comparison with previous
                                if ',' in content and '\n' in content:
                                    changes = compare_bitmaps(self.last_bitmap, content) if self.last_bitmap else []
                                    self.on_room_change_callback(content, formatted_content, changes)
                                    self.last_bitmap = content
                                else:
                                    self.on_room_change_callback(content, formatted_content, [])
                            except Exception as e:
                                print(f"Error in callback function: {e}")
                        elif ',' in content and '\n' in content:
                            # Store bitmap even if no callback
                            self.last_bitmap = content
                        
                        self.last_content = content
                    
                    self.last_modified = current_modified
                
                time.sleep(0.5)  # Check every 500ms
                
            except KeyboardInterrupt:
                print("\nMonitoring stopped.")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                time.sleep(1)



def on_room_change(raw_content, formatted_content, changes=[]):
    """
    Enhanced callback function that receives bitmap changes.
    
    Args:
        raw_content (str): The raw content from the save file
        formatted_content (str): The formatted/parsed content  
        changes (list): List of (old_value, new_value, position) tuples
    """
    print("🏠 ROOM CHANGED! Custom callback triggered.")
    
    # Check if the first line is already "1" - if so, skip processing
    if raw_content and raw_content.startswith("1"):
        print("🚫 First line is already '1' - skipping processing to avoid infinite loop")
        return
    
    # VAE Interpolation Processing
    if ',' in raw_content and '\n' in raw_content:
        print("\n🤖 VAE INTERPOLATION PROCESSING:")
        print("-" * 50)
        
        try:
            # Load trained VAE model with multiple path options
            script_dir = Path(__file__).parent if __file__ else Path.cwd()
            model_paths = [
                script_dir / "Data" / "best.pt",
                Path("Data/best.pt"),
                Path("./Data/best.pt"),
                script_dir / "best.pt"
            ]
            
            model_path = None
            for path in model_paths:
                if path.exists():
                    model_path = path
                    break
            
            if model_path:
                print(f"🔥 Loading VAE model from {model_path}")
                model = load_model(str(model_path))
                
                # Load random mutation bitmap
                random_bitmap = load_random_mutation_bitmap()
                
                if random_bitmap is not None:
                    # Print raw content and random bitmap for debugging
                    print("\n📋 RAW CONTENT (Current Room):")
                    print("=" * 50)
                    print(raw_content)
                    print("=" * 50)
                    
                    print("\n🎲 RANDOM MUTATION BITMAP:")
                    print("=" * 50)
                    print(random_bitmap)
                    print("=" * 50)
                    # Generate random alpha between 0.2 and 1
                    alpha = random.uniform(0.2, 1)
                    print(f"🎲 Random alpha: {alpha:.3f}")
                    
                    # Interpolate between current room and random mutation
                    interpolated_array = interpolate_bitmaps_with_alpha(
                        model, raw_content, random_bitmap, alpha
                    )
                    
                    if interpolated_array is not None:
                        print(f"✅ Successfully interpolated bitmaps!")
                        print(f"📊 Interpolated array shape: {interpolated_array.shape}")
                        
                        # Convert back to bitmap string format
                        interpolated_bitmap = convert_array_to_bitmap_string(interpolated_array)
                        
                        # Print the generated bitmap in console
                        print("\n🎨 GENERATED INTERPOLATED BITMAP:")
                        print("=" * 60)
                        print_bitmap_visual(interpolated_bitmap)
                        print("=" * 60)
                        
                        # Use interpolated bitmap instead of CONSTANT_BITMAP for comparison
                        print("\n🔍 Comparing with INTERPOLATED BITMAP:")
                        constant_changes = compare_bitmaps(raw_content, interpolated_bitmap)
                        
                        if constant_changes:
                            print(f"📊 Found {len(constant_changes)} differences from interpolated bitmap:")
                            print_bitmap_changes(constant_changes)
                            
                            # Generate Isaac mod code for the interpolated bitmap changes
                            grid_width = len(raw_content.split('\n')[0].split(','))
                            print("\n💻 Generated Isaac mod code (vs INTERPOLATED BITMAP):")
                            print("-" * 40)
                            
                            # Prepare changes data to write to file
                            changes_data = []
                            for old_val, new_val, pos in constant_changes:
                                row, col = pos
                                grid_index = get_grid_index_from_position(row, col, grid_width)
                                print(f"ProcessBitmapChange(room, {grid_index}, {old_val}, {new_val}, entity)")
                                changes_data.append(f"{old_val}, {new_val}, {grid_index}")
                            print("-" * 40)
                            
                            # Write changes to save1.dat file
                            try:
                                save_file_path = "../../../../../data/mutationmodel/save1.dat"
                                with open(save_file_path, 'a', encoding='utf-8') as file:
                                    file.write("\n")  # Add newline after bitmap
                                    for change_line in changes_data:
                                        file.write(f"{change_line}\n")
                                
                                # Read the current content
                                with open(save_file_path, 'r', encoding='utf-8') as file:
                                    current_content = file.read()
                                
                                # Write "1" at the top, then the rest of the content
                                with open(save_file_path, 'w', encoding='utf-8') as file:
                                    file.write("1\n" + current_content)
                                
                                print(f"✅ Written {len(changes_data)} interpolated changes to {save_file_path}")
                                print(f"✅ Added '1' signal at the top of the file")
                                print(f"🎯 Alpha used: {alpha:.3f}")
                            except Exception as e:
                                print(f"❌ Error writing to save file: {e}")
                        else:
                            print("✅ New bitmap matches interpolated bitmap exactly!")
                            

                            # Write "1" signal even when no changes are needed
                            try:
                                save_file_path = "../../../../../data/mutationmodel/save1.dat"
                                
                                # Read the current content
                                with open(save_file_path, 'r', encoding='utf-8') as file:
                                    current_content = file.read()
                                
                                # Write "1" at the top, then the rest of the content
                                with open(save_file_path, 'w', encoding='utf-8') as file:
                                    file.write("1\n" + current_content)
                                
                                print("✅ Added '1' signal (no interpolated changes needed)")
                                print(f"🎯 Alpha used: {alpha:.3f}")
                            except Exception as e:
                                print(f"❌ Error writing to save file: {e}")
                    else:
                        print("❌ Failed to interpolate bitmaps")
                        # Fallback to original CONSTANT_BITMAP comparison
                        fallback_to_constant_bitmap(raw_content)
                else:
                    print("❌ No random mutation bitmap available")
                    # Fallback to original CONSTANT_BITMAP comparison
                    fallback_to_constant_bitmap(raw_content)
            else:
                print(f"❌ VAE model not found at {model_path}")
                # Fallback to original CONSTANT_BITMAP comparison
                fallback_to_constant_bitmap(raw_content)
                
        except Exception as e:
            print(f"❌ Error during VAE processing: {e}")
            # Fallback to original CONSTANT_BITMAP comparison
            fallback_to_constant_bitmap(raw_content)
    
    if changes:
        print("\n📝 Changes from previous bitmap:")
        print_bitmap_changes(changes)
    
    return

def fallback_to_constant_bitmap(raw_content):
    """Fallback function to use original CONSTANT_BITMAP comparison"""
    print("\n🔄 Falling back to CONSTANT_BITMAP comparison:")
    
    constant_changes = compare_bitmaps(raw_content, CONSTANT_BITMAP)
    
    if constant_changes:
        print(f"📊 Found {len(constant_changes)} differences from CONSTANT_BITMAP:")
        print_bitmap_changes(constant_changes)
        
        # Generate Isaac mod code for the constant bitmap changes
        grid_width = len(raw_content.split('\n')[0].split(','))
        print("\n💻 Generated Isaac mod code (vs CONSTANT_BITMAP):")
        print("-" * 40)
        
        # Prepare changes data to write to file
        changes_data = []
        for old_val, new_val, pos in constant_changes:
            row, col = pos
            grid_index = get_grid_index_from_position(row, col, grid_width)
            print(f"ProcessBitmapChange(room, {grid_index}, {old_val}, {new_val}, entity)")
            changes_data.append(f"{old_val}, {new_val}, {grid_index}")
        print("-" * 40)
        
        # Write changes to save1.dat file
        try:
            save_file_path = "../../../../../data/mutationmodel/save1.dat"
            with open(save_file_path, 'a', encoding='utf-8') as file:
                file.write("\n")  # Add newline after bitmap
                for change_line in changes_data:
                    file.write(f"{change_line}\n")
            
            # Read the current content
            with open(save_file_path, 'r', encoding='utf-8') as file:
                current_content = file.read()
            
            # Write "1" at the top, then the rest of the content
            with open(save_file_path, 'w', encoding='utf-8') as file:
                file.write("1\n" + current_content)
            
            print(f"✅ Written {len(changes_data)} changes to {save_file_path}")
            print("✅ Added '1' signal at the top of the file")
        except Exception as e:
            print(f"❌ Error writing to save file: {e}")
    else:
        print("✅ New bitmap matches CONSTANT_BITMAP exactly!")
        
        # Write "1" signal even when no changes are needed
        try:
            save_file_path = "../../../../../data/mutationmodel/save1.dat"
            
            # Read the current content
            with open(save_file_path, 'r', encoding='utf-8') as file:
                current_content = file.read()
            
            # Write "1" at the top, then the rest of the content
            with open(save_file_path, 'w', encoding='utf-8') as file:
                file.write("1\n" + current_content)
            
            print("✅ Added '1' signal at the top of the file (no changes needed)")
        except Exception as e:
            print(f"❌ Error writing to save file: {e}")

def compare_bitmaps(old_bitmap, new_bitmap):
    """
    Compare two bitmap strings and return a list of changes.
    
    Args:
        old_bitmap (str): The original bitmap string (comma-separated values with newlines)
        new_bitmap (str): The new bitmap string to compare against
    
    Returns:
        list: List of tuples (old_value, new_value, position) where position is (row, col)
              Returns empty list if bitmaps are identical or invalid
    """
    changes = []
    
    if not old_bitmap or not new_bitmap:
        print("Error: One or both bitmaps are empty")
        return changes
    
    try:
        # Parse old bitmap
        old_lines = [line.strip() for line in old_bitmap.split('\n') if line.strip()]
        old_grid = []
        for line in old_lines:
            old_grid.append(line.split(','))
        
        # Parse new bitmap
        new_lines = [line.strip() for line in new_bitmap.split('\n') if line.strip()]
        new_grid = []
        for line in new_lines:
            new_grid.append(line.split(','))
        
        # Check if dimensions match
        if len(old_grid) != len(new_grid):
            print(f"Error: Bitmap height mismatch. Old: {len(old_grid)}, New: {len(new_grid)}")
            return changes
        
        # Compare each cell
        for row in range(len(old_grid)):
            if len(old_grid[row]) != len(new_grid[row]):
                print(f"Error: Row {row} width mismatch. Old: {len(old_grid[row])}, New: {len(new_grid[row])}")
                continue
            
            for col in range(len(old_grid[row])):
                old_value = old_grid[row][col].strip()
                new_value = new_grid[row][col].strip()
                
                # Skip border cells (values 0=WALL and 1=DOOR)
                if old_value in ['0', '1'] and new_value in ['0', '1']:
                    continue
                
                if old_value != new_value:
                    changes.append((old_value, new_value, (row, col)))
        
    except Exception as e:
        print(f"Error comparing bitmaps: {e}")
        return []
    
    return changes

def print_bitmap_changes(changes):
    """
    Print the bitmap changes in a readable format.
    
    Args:
        changes (list): List of tuples (old_value, new_value, position)
    """
    if not changes:
        print("No changes detected between bitmaps.")
        return
    
    # Tile type names for better readability
    tile_names = {
        "0": "WALL", "1": "DOOR", "2": "FREE_SPACE", "3": "STONE",
        "4": "PIT", "5": "BLOCK", "6": "ENTITY", "7": "PICKUP",
        "8": "MACHINE", "9": "FIRE", "10": "POOP", "11": "SPIKE"
    }
    
    print(f"\n🔄 Found {len(changes)} changes:")
    print("-" * 60)
    
    for i, (old_val, new_val, pos) in enumerate(changes, 1):
        row, col = pos
        old_name = tile_names.get(old_val, f"UNKNOWN({old_val})")
        new_name = tile_names.get(new_val, f"UNKNOWN({new_val})")
        
        print(f"{i:2d}. Position ({row:2d},{col:2d}): {old_name:10s} → {new_name:10s} ({old_val} → {new_val})")
    
    print("-" * 60)

def get_grid_index_from_position(row, col, grid_width):
    """
    Convert 2D position to 1D grid index (like Isaac uses).
    
    Args:
        row (int): Row position (0-based)
        col (int): Column position (0-based)
        grid_width (int): Width of the grid
    
    Returns:
        int: Grid index
    """
    return row * grid_width + col

def bitmap_to_array_without_border(bitmap_content):
    """
    Convert a 15x9 bitmap string to a 13x7 numpy array by removing the outer border.
    
    Args:
        bitmap_content (str): The bitmap string (comma-separated values with newlines)
    
    Returns:
        np.ndarray: 13x7 numpy array with the border removed, or None if invalid
    """
    try:
        # Parse bitmap into lines
        lines = [line.strip() for line in bitmap_content.split('\n') if line.strip()]
        
        # Convert to 2D list
        grid = []
        for line in lines:
            row = [int(val.strip()) for val in line.split(',')]
            grid.append(row)
        
        # Convert to numpy array
        bitmap_array = np.array(grid)
        
        # Check if dimensions are correct (15x9)
        if bitmap_array.shape != (9, 15):
            print(f"Warning: Expected 9x15 bitmap, got {bitmap_array.shape}")
            return None
        
        # Remove outer border (remove first/last row and first/last column)
        # This converts 15x9 -> 13x7
        inner_array = bitmap_array[1:-1, 1:-1]
        
        print(f"Converted {bitmap_array.shape} bitmap to {inner_array.shape} array (border removed)")
        return inner_array
        
    except Exception as e:
        print(f"Error converting bitmap to array: {e}")
        return None

def encode_bitmap_content_to_latent(model, bitmap_content):
    """Convert raw bitmap content string to latent vector"""
    try:
        # Parse bitmap into lines and convert to numpy array
        lines = [line.strip() for line in bitmap_content.split('\n') if line.strip()]
        grid = []
        for line in lines:
            row = [int(val.strip()) for val in line.split(',')]
            grid.append(row)
        
        bitmap_array = np.array(grid)
        
        # Remove border if it's 15x9, keep if it's already 13x7
        if bitmap_array.shape == (9, 15):
            # Remove outer border: 15x9 -> 13x7
            inner_array = bitmap_array[1:-1, 1:-1]
        elif bitmap_array.shape == (7, 13):
            inner_array = bitmap_array
        else:
            print(f"Warning: Unexpected bitmap shape {bitmap_array.shape}")
            return None
        
        # Convert to tensor and encode
        x = torch.from_numpy(inner_array).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        
        with torch.no_grad():
            mu, logvar = model.encode(x)
            return mu
            
    except Exception as e:
        print(f"Error encoding bitmap content: {e}")
        return None

def load_random_mutation_bitmap():
    """Load a random bitmap from Mutations_Extracted folder and convert to entity values"""
    # Try multiple paths for the mutations data
    script_dir = Path(__file__).parent if __file__ else Path.cwd()
    mutation_paths = [
        script_dir / "Data" / "Mutations_Extracted",
        Path("Data/Mutations_Extracted"),
        Path("./Data/Mutations_Extracted"),
        script_dir / "Mutations_Extracted"
    ]
    
    mutations_path = None
    for path in mutation_paths:
        if path.exists():
            mutations_path = path
            break
    
    if not mutations_path:
        print(f"Warning: Mutations_Extracted folder not found in any of these locations:")
        for path in mutation_paths:
            print(f"  - {path}")
        return None
    
    # Find all .bmp files recursively
    bmp_files = list(mutations_path.rglob("*.bmp"))
    
    if not bmp_files:
        print(f"No .bmp files found in {mutations_path}")
        return None
    
    # Select random file
    random_file = random.choice(bmp_files)
    print(f"🎯 Selected random mutation: {random_file.name}")
    print(f"📁 Full path: {random_file}")
    
    try:
        # Load image
        img = Image.open(random_file).convert("L")
        arr = np.array(img)
        
        print(f"📊 Image dimensions: {arr.shape}")
        print(f"📊 Pixel value range: {arr.min()} - {arr.max()}")
        print(f"📊 Unique pixel values: {np.unique(arr)}")
        
        # Create TBoI_Bitmap instance for conversion
        tboi_bitmap = TBoI_Bitmap()
        
        # Convert pixel values to entity values
        entity_arr = np.zeros_like(arr)
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                pixel_value = arr[y, x]
                try:
                    # Convert pixel value to EntityType using tboi_bitmap logic
                    entity_type = tboi_bitmap.get_entity_id_with_pixel_value(pixel_value)
                    entity_arr[y, x] = entity_type.value
                except:
                    # Fallback to FREE_SPACE if conversion fails
                    entity_arr[y, x] = EntityType.FREE_SPACE.value
        
        print(f"📊 Converted entity value range: {entity_arr.min()} - {entity_arr.max()}")
        print(f"📊 Unique entity values: {np.unique(entity_arr)}")
        
        # Convert to comma-separated string format
        bitmap_content = ""
        for row in entity_arr:
            row_str = ",".join(str(int(val)) for val in row)
            bitmap_content += row_str + "\n"
        
        return bitmap_content.strip()
        
    except Exception as e:
        print(f"Error loading random mutation bitmap: {e}")
        return None

def interpolate_bitmaps_with_alpha(model, bitmap1_content, bitmap2_content, alpha):
    """Interpolate between two bitmap contents with given alpha"""
    try:
        # Encode both bitmaps to latent space
        z1 = encode_bitmap_content_to_latent(model, bitmap1_content)
        z2 = encode_bitmap_content_to_latent(model, bitmap2_content)
        
        if z1 is None or z2 is None:
            print("Failed to encode one or both bitmaps")
            return None
        
        # Interpolate: z = (1-α)*z1 + α*z2
        z_interp = (1 - alpha) * z1 + alpha * z2
        
        # Decode interpolated latent vector
        with torch.no_grad():
            output_logits = model.decode(z_interp)
            pred = torch.argmax(output_logits, dim=1)[0].cpu().numpy()
        
        return pred
        
    except Exception as e:
        print(f"Error during interpolation: {e}")
        return None

def convert_array_to_bitmap_string(arr):
    """Convert 7x13 numpy array to bitmap string format"""
    # Add border to make it 9x15
    bordered_arr = np.zeros((9, 15), dtype=int)
    bordered_arr[1:-1, 1:-1] = arr  # Place 7x13 array in center
    
    # Convert to string format
    bitmap_content = ""
    for row in bordered_arr:
        row_str = ",".join(str(val) for val in row)
        bitmap_content += row_str + "\n"
    
    return bitmap_content.strip()

def print_bitmap_visual(bitmap_content):
    """
    Print the bitmap in a visual format with tile type symbols.
    
    Args:
        bitmap_content (str): The bitmap string (comma-separated values with newlines)
    """
    # Tile symbols for visual representation
    tile_symbols = {
        "0": "█",  # WALL - solid block
        "1": "▒",  # DOOR - light shade
        "2": "·",  # FREE_SPACE - small dot
        "3": "▓",  # STONE - medium shade
        "4": "○",  # PIT - circle
        "5": "■",  # BLOCK - square
        "6": "♦",  # ENTITY - diamond
        "7": "♥",  # PICKUP - heart
        "8": "♠",  # MACHINE - spade
        "9": "▲",  # FIRE - triangle
        "10": "♣", # POOP - club
        "11": "↑"  # SPIKE - up arrow
    }
    
    tile_names = {
        "0": "WALL", "1": "DOOR", "2": "FREE", "3": "STONE",
        "4": "PIT", "5": "BLOCK", "6": "ENTITY", "7": "PICKUP",
        "8": "MACHINE", "9": "FIRE", "10": "POOP", "11": "SPIKE"
    }
    
    try:
        lines = [line.strip() for line in bitmap_content.split('\n') if line.strip()]
        
        print("Visual representation:")
        print("    " + "".join([f"{i%10}" for i in range(15)]))  # Column numbers
        print("   ┌" + "─" * 15 + "┐")
        
        for row_idx, line in enumerate(lines):
            values = [val.strip() for val in line.split(',')]
            symbols = [tile_symbols.get(val, "?") for val in values]
            print(f"{row_idx:2d} │{''.join(symbols)}│")
        
        print("   └" + "─" * 15 + "┘")
        
        # Print legend
        print("\nLegend:")
        legend_items = []
        for val, symbol in tile_symbols.items():
            name = tile_names.get(val, f"TYPE_{val}")
            legend_items.append(f"{symbol}={name}")
        
        # Print legend in rows of 4
        for i in range(0, len(legend_items), 4):
            row_items = legend_items[i:i+4]
            print("  " + "  ".join(f"{item:12s}" for item in row_items))
        
        # Print raw data
        print(f"\nRaw bitmap data:")
        print("-" * 30)
        for i, line in enumerate(lines):
            print(f"Row {i:2d}: {line}")
        
    except Exception as e:
        print(f"Error displaying bitmap: {e}")
        print("Raw content:")
        print(bitmap_content)

def main():
    # Path to the save file
    save_file_path = "../../../../../data/mutationmodel/save1.dat"
    
    # Create monitor instance with callback
    monitor = SaveFileMonitor(save_file_path, on_room_change_callback=on_room_change)
    
    # Check if file exists initially
    if not monitor.check_file_exists():
        print(f"Save file not found: {save_file_path}")
        print("Make sure the path is correct and the game has created the save file.")
        print("The script will wait for the file to be created...")
    
    # Start monitoring
    monitor.monitor()

if __name__ == "__main__":
    main()
