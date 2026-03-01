#!/usr/bin/env python3
"""Convert unsafe get_unchecked patterns in simd_trackers.rs to use conditional macros."""
import re

FILE = "src/objective_tracker_impl/simd_trackers.rs"

with open(FILE) as f:
    content = f.read()

original = content

# Pattern 1: Single-expression let bindings with get_unchecked (value read)
# e.g.: let start = unsafe { *self.clear_elements_offsets.get_unchecked(image_index) };
content = re.sub(
    r'unsafe \{ \*self\.(\w+)\.get_unchecked\(([^)]+)\) \}',
    r'unchecked_get!(self.\1, \2)',
    content
)

# Pattern 2: Slice get_unchecked (range)  
# e.g.: let clear_elements = unsafe { self.clear_elements.get_unchecked(start..end) };
content = re.sub(
    r'unsafe \{ self\.(\w+)\.get_unchecked\((\w+\.\.\w+)\) \}',
    r'unchecked_slice!(self.\1, \2)',
    content
)

# Pattern 3: get_unchecked_mut returning mutable reference
# e.g.: let slot = self.element_packed_small.get_unchecked_mut(idx);
# These appear inside unsafe blocks, so we need to handle them there
content = re.sub(
    r'self\.(\w+)\.get_unchecked_mut\(([^)]+)\)',
    r'unchecked_get_mut!(self.\1, \2)',
    content
)

# Pattern 4: Deref of get_unchecked inside expressions
# e.g.: (*self.counts.get_unchecked(idx) == 1)
content = re.sub(
    r'\(\*self\.(\w+)\.get_unchecked\(([^)]+)\)',
    r'(unchecked_get!(self.\1, \2)',
    content
)

# Pattern 4b: Standalone *self.XXX.get_unchecked(YYY) not already caught
content = re.sub(
    r'\*self\.(\w+)\.get_unchecked\(([^)]+)\)',
    r'unchecked_get!(self.\1, \2)',
    content
)

with open(FILE, 'w') as f:
    f.write(content)

# Count changes
import difflib
orig_lines = original.splitlines()
new_lines = content.splitlines()
diff = list(difflib.unified_diff(orig_lines, new_lines, lineterm=''))
adds = sum(1 for l in diff if l.startswith('+') and not l.startswith('+++'))
removes = sum(1 for l in diff if l.startswith('-') and not l.startswith('---'))
print(f"Changed {removes} lines -> {adds} lines")
print(f"Remaining 'unsafe' occurrences:")
import subprocess
subprocess.run(["grep", "-c", "unsafe", FILE])
subprocess.run(["grep", "-n", "unsafe", FILE])
