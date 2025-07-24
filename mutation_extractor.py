import os
import shutil
from pathlib import Path
import re

def extract_every_20th_mutation(source_dir="Bitmaps/Mutations", output_dir="Bitmaps/Mutations_Extracted", interval=20):
    """
    Extracts every 20th mutation from each bitmap_* folder.
    
    Args:
        source_dir: Path to the mutations folder
        output_dir: Path where extracted mutations will be saved
        interval: Extract every Nth mutation (default: 20)
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_extracted = 0
    
    # Go through each bitmap_* folder
    for bitmap_folder in sorted(source_path.iterdir()):
        if not bitmap_folder.is_dir() or not bitmap_folder.name.startswith("bitmap_"):
            continue
            
        print(f"📁 Processing {bitmap_folder.name}...")
        
        # Create corresponding output folder
        output_bitmap_folder = output_path / bitmap_folder.name
        output_bitmap_folder.mkdir(exist_ok=True)
        
        # Get all mutation files and sort them
        mutation_files = []
        for file in bitmap_folder.glob("mutation_*.bmp"):
            # Extract mutation number from filename
            match = re.match(r'mutation_(\d+)', file.stem)
            if match:
                mutation_num = int(match.group(1))
                mutation_files.append((mutation_num, file))
        
        # Sort by mutation number
        mutation_files.sort(key=lambda x: x[0])
        
        if not mutation_files:
            continue
        
        # NEW: Extract mutations where mutation_number % interval == 0
        extracted_count = 0
        for mutation_num, source_file in mutation_files:
            if mutation_num % interval == 0:  # 0, 20, 40, 60, etc.
                # Copy to output folder
                dest_file = output_bitmap_folder / source_file.name
                shutil.copy2(source_file, dest_file)
                
                extracted_count += 1
                total_extracted += 1
                
                print(f"  ✅ Extracted mutation_{mutation_num} -> {dest_file.name}")
        
        print(f"  📊 Extracted {extracted_count} mutations from {len(mutation_files)} total")
    
    print(f"\n🎯 SUMMARY:")
    print(f"   Total mutations extracted: {total_extracted}")
    print(f"   Output directory: {output_dir}")
    print(f"   Interval: every {interval}th mutation")

def extract_specific_mutations(source_dir="Bitmaps/Mutations", output_dir="Bitmaps/Mutations_Specific", 
                             mutation_numbers=[0, 20, 40, 60, 80]):
    """
    Extracts specific mutation numbers from each bitmap folder.
    
    Args:
        source_dir: Path to the mutations folder
        output_dir: Path where extracted mutations will be saved
        mutation_numbers: List of specific mutation numbers to extract
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return
    
    output_path.mkdir(parents=True, exist_ok=True)
    total_extracted = 0
    
    for bitmap_folder in sorted(source_path.iterdir()):
        if not bitmap_folder.is_dir() or not bitmap_folder.name.startswith("bitmap_"):
            continue
            
        print(f"📁 Processing {bitmap_folder.name}...")
        
        output_bitmap_folder = output_path / bitmap_folder.name
        output_bitmap_folder.mkdir(exist_ok=True)
        
        extracted_count = 0
        
        for target_num in mutation_numbers:
            # Look for mutation files with this number
            pattern = f"mutation_{target_num}_*"
            matching_files = list(bitmap_folder.glob(f"{pattern}.bmp"))
            
            if not matching_files:
                # Try exact match
                exact_file = bitmap_folder / f"mutation_{target_num}.bmp"
                if exact_file.exists():
                    matching_files = [exact_file]
            
            if matching_files:
                # Take the first matching file
                source_file = matching_files[0]
                dest_file = output_bitmap_folder / source_file.name
                shutil.copy2(source_file, dest_file)
                
                extracted_count += 1
                total_extracted += 1
                print(f"  ✅ Extracted {source_file.name}")
            else:
                print(f"  ⚠️  mutation_{target_num} not found")
        
        print(f"  📊 Extracted {extracted_count}/{len(mutation_numbers)} requested mutations")
    
    print(f"\n🎯 SUMMARY:")
    print(f"   Total mutations extracted: {total_extracted}")
    print(f"   Output directory: {output_dir}")
    print(f"   Requested mutations: {mutation_numbers}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract mutations from bitmap folders")
    parser.add_argument("--source", default="Bitmaps/Mutations", 
                       help="Source mutations directory")
    parser.add_argument("--output", default="Bitmaps/Mutations_Extracted", 
                       help="Output directory")
    parser.add_argument("--interval", type=int, default=20, 
                       help="Extract every Nth mutation")
    parser.add_argument("--specific", nargs="+", type=int, 
                       help="Extract specific mutation numbers (e.g., --specific 0 20 40)")
    
    args = parser.parse_args()
    
    if args.specific:
        print(f"🎯 Extracting specific mutations: {args.specific}")
        extract_specific_mutations(args.source, args.output, args.specific)
    else:
        print(f"🎯 Extracting every {args.interval}th mutation")
        extract_every_20th_mutation(args.source, args.output, args.interval)

if __name__ == "__main__":
    main()