from typing import Tuple


def bytes_to_readable(b: Tuple[bytes, ...]) -> str:
    """
    Convert a tuple of bytes to readable format.
    
    Examples:
        (b'm', b'o', b're') → 'm|o|re'
        (b'r', b'e') → 'r|e'
        (b' ', b'a') → ' |a'
    """
    def decode_single(byte_val: bytes) -> str:
        try:
            decoded = byte_val.decode('utf-8')
            if decoded == '\n': return '<newline>'
            elif decoded == '\t': return '<tab>'
            elif decoded == '\r': return '<return>'
            else: return decoded
        except UnicodeDecodeError:
            return f"0x{byte_val.hex()}"
    
    return "|".join(decode_single(b_val) for b_val in b)


def format_byte_pairs(byte_pair_freqs: dict, top_n: int = 10) -> str:
    """
    Format byte pair frequencies in a readable way.
    
    Args:
        byte_pair_freqs: Dictionary mapping byte pairs to frequencies
        top_n: Number of top pairs to display
    
    Returns:
        Formatted string with top byte pairs
    """
    if not byte_pair_freqs:
        return "No byte pairs"
    
    top_pairs = sorted(byte_pair_freqs.items(), key=lambda x: (-x[1], x[0]))[:top_n]
    lines = []
    
    for pair, count in top_pairs:
        left = bytes_to_readable((pair[0],))
        right = bytes_to_readable((pair[1],))
        readable = f"{left}|{right}"
        lines.append(f"  {readable}: {count}")
    
    return "\n".join(lines)