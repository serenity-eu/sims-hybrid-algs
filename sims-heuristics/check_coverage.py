import json

# Load the problem data from sims-problem
import sys
sys.path.insert(0, '../sims-problem/tests/data')
from lagos_nigeria_30 import PROBLEM

# Check how many images cover each element
coverage_counts = [0] * PROBLEM.num_elements

for img_idx in range(PROBLEM.num_images):
    for elem_idx in PROBLEM.images[img_idx]:
        coverage_counts[elem_idx] += 1

print(f"Total elements: {PROBLEM.num_elements}")
print(f"Total images: {PROBLEM.num_images}")
print(f"Min images covering an element: {min(coverage_counts)}")
print(f"Max images covering an element: {max(coverage_counts)}")
print(f"Average images per element: {sum(coverage_counts) / len(coverage_counts):.2f}")
print(f"\nElements covered by > 16 images: {sum(1 for c in coverage_counts if c > 16)}")
print(f"Elements covered by > 8 images: {sum(1 for c in coverage_counts if c > 8)}")
print(f"\nDistribution:")
for threshold in [1, 2, 4, 8, 16, 32]:
    count = sum(1 for c in coverage_counts if c >= threshold)
    pct = 100*count/len(coverage_counts) if coverage_counts else 0
    print(f"  >= {threshold:2d} images: {count:5d} elements ({pct:.1f}%)")
