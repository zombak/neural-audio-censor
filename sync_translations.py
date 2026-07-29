import os
import glob
from pathlib import Path
from typing import Dict, Set

def load_translations(filepath: str) -> Dict[str, str]:
    """Loads translations from a file into a dictionary."""
    translations = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and "=" in line:
                    k, v = line.split("=", 1)
                    translations[k.strip()] = v.strip()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return translations

def sync_languages(reference_file: str = "ui-english.txt"):
    """Synchronizes all translation files with the reference file."""
    if not os.path.exists(reference_file):
        print(f"❌ Error: Reference file {reference_file} not found!")
        return

    print(f"📖 Loading reference from {reference_file}...")
    master_translations = load_translations(reference_file)
    master_keys = set(master_translations.keys())
    
    # Find all files matching the pattern ui-*.txt
    translation_files = glob.glob("ui-*.txt")
    
    if not translation_files:
        print("No translation files found.")
        return

    total_updates = 0

    for file_path in translation_files:
        # Skip the reference file itself
        if file_path == reference_file:
            continue
            
        current_translations = load_translations(file_path)
        current_keys = set(current_translations.keys())
        
        # Identify keys present in the reference but missing in the current file
        missing_keys = master_keys - current_keys
        
        if missing_keys:
            print(f"Updating {file_path}: missing {len(missing_keys)} keys...")
            
            with open(file_path, "a", encoding="utf-8") as f:
                # Add a separator comment for clarity
                f.write("\n# Added by sync_translations.py\n")
                for key in missing_keys:
                    default_val = master_translations[key]
                    f.write(f"{key}={default_val}\n")
            
            total_updates += len(missing_keys)
            for mk in missing_keys:
                print(f"  + {mk}")
        else:
            print(f"✅ {file_path} is up to date.")

    print("-" * 30)
    if total_updates > 0:
        print(f"🚀 Synchronization complete. Added {total_updates} strings.")
        print("Please open the updated files and replace the English placeholders with actual translations.")
    else:
        print("All translation files are synchronized. No changes were required.")

if __name__ == "__main__":
    sync_languages()
