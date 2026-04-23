def build_logger_name(name: str) -> str:
    """
    Generates a hierarchical logger name based on the file's location within the repository.
    """
    from pathlib import Path
    # Convert name to hierarchical format if it's a file path
    if isinstance(name, str):
        # If it looks like a python file or path, convert to Path object
        if name.endswith('.py') or "/" in name or "\\" in name:
            name = Path(name)
    

    path = Path(name).resolve()
    target = "kon-artist"  # Adjust if your repo has a different name or structure
    
    parts = path.parts
    if target in parts:
        # Remove +1 if you want repo root included in the string
        idx = parts.index(target) + 1 
        
        # Slice the parts from that index onwards
        relevant_parts = parts[idx:]
        
        # Reconstruct path to safely remove the extension
        clean_path = Path(*relevant_parts).with_suffix('')
        
        # Join with dots
        return ".".join(clean_path.parts)
        # Returns: core.utils.data_io

    # Fallback if running outside the repo structure
    return Path(name).stem

def create_timestamp() -> str:
    """Generates a timestamp string for logging and checkpointing."""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")