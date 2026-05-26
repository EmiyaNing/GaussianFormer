import torch
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

print("Testing basic imports...")
try:
    from local_aggregate import LocalAggregator
    print("✓ Import successful!")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

print("\nTesting class instantiation...")
try:
    aggregator = LocalAggregator(3, 200, 200, 16, [-40.0, -40.0, -1.0], 0.4)
    print("✓ Instantiation successful!")
except Exception as e:
    print(f"✗ Instantiation failed: {e}")
    sys.exit(1)

print("\nChecking if inverse_render method exists...")
if hasattr(aggregator, 'inverse_render'):
    print("✓ inverse_render method exists!")
else:
    print("✗ inverse_render method does NOT exist!")
    print("Available methods:", [method for method in dir(aggregator) if not method.startswith('_')])
    sys.exit(1)

print("\nTesting with small data...")
try:
    P = 100  # 小批量
    N = 200  # 小批量
    C = 18
    
    # 生成测试数据
    pts = torch.rand(1, N, 3)
    means3D = torch.rand(1, P, 3)
    opas = torch.rand(1, P)
    occupancy_gt = torch.rand(1, N, C)
    scales = torch.rand(1, P, 3)
    cov3D = torch.rand(1, P, 3, 3)
    
    print('Testing inverse_render function...')
    gaussian_semantic_mask = aggregator.inverse_render(pts, means3D, opas, occupancy_gt, scales, cov3D)
    print(f'✓ Output shape: {gaussian_semantic_mask.shape}')
    print('✓ Inverse render function works correctly!')
    
except Exception as e:
    print(f"✗ Inverse render test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 All tests passed!")