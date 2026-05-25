#!/usr/bin/env python3
import sys, os, struct
from collections import defaultdict
sys.path.insert(0, '/opt/homebrew/lib/python3.9/site-packages')
from gguf import GGUFReader

src = '/Users/ryantsudek/Projects/pma2-ltx-video/models/ltx-2.3-gguf/ltx-2.3-22b-Q4_K_M.gguf'
dst = '/Users/ryantsudek/Projects/pma2-ltx-video/models/ltx-2.3-gguf/ltx-2.3-22b-Q4_K_M-fixed.gguf'

print("Reading GGUF...")
reader = GGUFReader(src, 'r')
tensors = [{'name': t.name, 'shape': list(t.shape), 'data_offset': t.data_offset, 'n_bytes': t.n_bytes} for t in reader.tensors]

meta_start = 4544
meta_end = tensors[0]['data_offset']

with open(src, 'rb') as f:
    f.seek(meta_start)
    meta = bytearray(f.read(meta_end - meta_start))

# Parse entries
entries = []
off = 0

for i in range(len(tensors)):
    name_len = struct.unpack_from('<I', meta, off)[0]
    name_start = off + 8
    name = meta[name_start:name_start + name_len].decode('utf-8', errors='replace')
    n_dims_off = name_start + name_len
    n_dims = struct.unpack_from('<I', meta, n_dims_off)[0]
    dims_off = n_dims_off + 4
    dims = list(struct.unpack_from(f'<{n_dims}Q', meta, dims_off))
    dtype_off = dims_off + n_dims * 8
    dtype = struct.unpack_from('<I', meta, dtype_off)[0]
    offset_off = dtype_off + 4
    offset = struct.unpack_from('<Q', meta, offset_off)[0]
    entry_end = offset_off + 8
    
    entries.append({'tensor_idx': i, 'meta_start': off, 'meta_end': entry_end,
                   'name': name, 'name_len': name_len, 'n_dims': n_dims,
                   'dims': dims, 'dtype': dtype, 'offset': offset,
                   'new_name': None, 'shift': 0})
    off = entry_end

print(f"Parsed {len(entries)} entries")

# Handle long names (>= 64 chars) with collision detection
MAX_NAME = 64
used_new_names = set()
collision_groups = defaultdict(list)

for e in entries:
    if len(e['name']) >= MAX_NAME:
        truncated = e['name'][:61]  # Reserve 2 chars for disambiguation
        collision_groups[truncated].append(e)

# Disambiguate collisions with 2-char hex suffix
print(f"Collision groups: {len(collision_groups)}")
for truncated, group in collision_groups.items():
    for e in group:
        for suffix in [f'{i:02x}' for i in range(256)]:
            candidate = truncated + suffix
            if candidate not in used_new_names:
                used_new_names.add(candidate)
                e['new_name'] = candidate
                e['shift'] = e['name_len'] - len(candidate)
                break

# Verify uniqueness
final_names = [e['new_name'] if e['new_name'] else e['name'] for e in entries]
print(f"Total: {len(final_names)}, Unique: {len(set(final_names))}")

# Calculate adjustments
total_shift = sum(e['shift'] for e in entries)
new_meta_size = len(meta) - total_shift
new_meta_end = meta_start + new_meta_size
new_tensor_data_start = (new_meta_end + 31) // 32 * 32
padding = new_tensor_data_start - new_meta_end
offset_adjustment = new_tensor_data_start - meta_end

print(f"Total shift: {total_shift}, Offset adjustment: {offset_adjustment}")

# Write fixed file
print("Writing fixed GGUF...")
with open(src, 'rb') as f_in, open(dst, 'wb') as f_out:
    f_in.seek(0)
    f_out.write(f_in.read(meta_start))
    
    new_meta = bytearray()
    for e in entries:
        if e['new_name']:
            new_entry = bytearray()
            new_entry.extend(struct.pack('<I', len(e['new_name'])))
            new_entry.extend(b'\x00\x00\x00\x00')
            new_entry.extend(e['new_name'].encode('utf-8'))
            new_entry.extend(struct.pack('<I', e['n_dims']))
            for d in e['dims']:
                new_entry.extend(struct.pack('<Q', d))
            new_entry.extend(struct.pack('<I', e['dtype']))
            new_entry.extend(struct.pack('<Q', e['offset'] + offset_adjustment))
            new_meta.extend(new_entry)
        else:
            new_meta.extend(meta[e['meta_start']:e['meta_end']])
    
    f_out.write(new_meta)
    
    if padding > 0:
        f_out.write(b'\x00' * padding)
    
    f_in.seek(meta_end)
    while True:
        chunk = f_in.read(8 * 1024 * 1024)
        if not chunk:
            break
        f_out.write(chunk)

print(f"Done: {os.path.getsize(dst):,} bytes")

# Verify the fixed file loads
print("\nVerifying fixed GGUF...")
try:
    reader2 = GGUFReader(dst, 'r')
    print(f"SUCCESS: {len(reader2.tensors)} tensors loaded")
    long_names = [t.name for t in reader2.tensors if len(t.name) >= 64]
    print(f"Long names (>=64): {len(long_names)}")
    if long_names:
        for n in long_names[:3]:
            print(f"  {n[:50]}...")
except Exception as ex:
    print(f"FAILED: {ex}")