import re
BLOCK_DELIMITERS = r'[，。！？；：、,.!?;:\n\r]+'
SENT_END = ".!?。！？"

def split_into_blocks(text, delimiters=BLOCK_DELIMITERS):
    return [p.strip() for p in re.split(delimiters, text) if p.strip()]

def split_sentences(text):
    blocks, buf = [], []
    for ch in (text or "").strip():
        buf.append(ch)
        if ch in SENT_END:
            blocks.append("".join(buf).strip()); buf = []
    return blocks
