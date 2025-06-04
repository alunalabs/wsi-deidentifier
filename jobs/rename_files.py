#!/usr/bin/env python3

import csv
import os
import json
import argparse
from pathlib import Path

def load_mappings():
    """Load mappings from CSV file"""
    mappings = {}
    csv_file = 'filename_to_id_mapping.csv'
    
    if not os.path.exists(csv_file):
        print(f"❌ CSV file {csv_file} not found")
        return mappings
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['filename']
            extracted_id = row['extracted_id']
            if extracted_id:
                # Replace .png with .svs and replace spaces with dashes
                svs_filename = filename.replace('.png', '.svs')
                new_filename = extracted_id.replace(' ', '-') + '.svs'
                mappings[svs_filename] = new_filename
    
    return mappings

def save_rename_log(renamed_files, log_file='rename_log.json'):
    """Save a log of renamed files for undo purposes"""
    with open(log_file, 'w') as f:
        json.dump(renamed_files, f, indent=2)

def load_rename_log(log_file='rename_log.json'):
    """Load the rename log for undo purposes"""
    if not os.path.exists(log_file):
        print(f"❌ Rename log {log_file} not found")
        return {}
    
    with open(log_file, 'r') as f:
        return json.load(f)

def rename_files():
    """Rename SVS files based on extracted IDs"""
    mappings = load_mappings()
    
    if not mappings:
        print("❌ No mappings found")
        return False
    
    print(f"Found {len(mappings)} file mappings")
    
    renamed_files = {}
    successful_renames = 0
    
    for original_filename, new_filename in mappings.items():
        if os.path.exists(original_filename):
            try:
                if os.path.exists(new_filename):
                    print(f"⚠️  Target file {new_filename} already exists, skipping {original_filename}")
                    continue
                
                os.rename(original_filename, new_filename)
                renamed_files[new_filename] = original_filename
                successful_renames += 1
                print(f"✅ Renamed: {original_filename} → {new_filename}")
                
            except Exception as e:
                print(f"❌ Error renaming {original_filename}: {e}")
        else:
            print(f"⚠️  File {original_filename} not found")
    
    if renamed_files:
        save_rename_log(renamed_files)
        print(f"\n✅ Successfully renamed {successful_renames}/{len(mappings)} files")
        print("💾 Rename log saved to rename_log.json")
    else:
        print("\n❌ No files were renamed")
    
    return successful_renames > 0

def undo_rename():
    """Undo the file renaming using the log"""
    renamed_files = load_rename_log()
    
    if not renamed_files:
        print("❌ No rename log found or log is empty")
        return False
    
    print(f"Found {len(renamed_files)} files to restore")
    
    successful_undos = 0
    
    for new_filename, original_filename in renamed_files.items():
        if os.path.exists(new_filename):
            try:
                if os.path.exists(original_filename):
                    print(f"⚠️  Target file {original_filename} already exists, skipping {new_filename}")
                    continue
                
                os.rename(new_filename, original_filename)
                successful_undos += 1
                print(f"✅ Restored: {new_filename} → {original_filename}")
                
            except Exception as e:
                print(f"❌ Error restoring {new_filename}: {e}")
        else:
            print(f"⚠️  File {new_filename} not found")
    
    if successful_undos > 0:
        # Remove the log file after successful undo
        try:
            os.remove('rename_log.json')
            print("💾 Rename log removed")
        except:
            pass
        
        print(f"\n✅ Successfully restored {successful_undos}/{len(renamed_files)} files")
    else:
        print("\n❌ No files were restored")
    
    return successful_undos > 0

def preview_renames():
    """Preview what files would be renamed without actually doing it"""
    mappings = load_mappings()
    
    if not mappings:
        print("❌ No mappings found")
        return
    
    print(f"Preview of {len(mappings)} file renames:")
    print("-" * 60)
    
    for original_filename, new_filename in mappings.items():
        status = "✅" if os.path.exists(original_filename) else "❌"
        print(f"{status} {original_filename} → {new_filename}")

def main():
    parser = argparse.ArgumentParser(description="Rename SVS files based on extracted IDs")
    parser.add_argument('action', choices=['rename', 'undo', 'preview'], 
                       help='Action to perform: rename files, undo renames, or preview changes')
    
    args = parser.parse_args()
    
    if args.action == 'rename':
        rename_files()
    elif args.action == 'undo':
        undo_rename()
    elif args.action == 'preview':
        preview_renames()

if __name__ == "__main__":
    main()