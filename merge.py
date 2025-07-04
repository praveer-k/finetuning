import json
from pathlib import Path

def merge_json_files(folder_path, output_file):
    """
    Merge all JSON files in a folder into a single JSON file.
    Each JSON file should contain an array of conversation objects.
    
    Args:
        folder_path: Path to folder containing JSON files
        output_file: Path for merged output file
    
    Returns:
        List of all merged conversations
    """
    merged_conversations = []
    json_files = list(Path(folder_path).glob("*.json"))
    
    print(f"Found {len(json_files)} JSON files to merge")
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
                merged_conversations.extend(conversations)
                print(f"✓ Merged {len(conversations)} messages from {file_path.name}")
        except Exception as e:
            print(f"✗ Error loading {file_path.name}: {e}")
    
    # Save merged file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_conversations, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Successfully merged {len(merged_conversations)} messages into {output_file}")
    return merged_conversations

merge_json_files("./.", "./merged_conversations.json")
