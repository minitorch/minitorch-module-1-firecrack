from minitorch import Scalar, topological_sort
from project.run_scalar import Linear

layer = Linear(2, 5)
y = layer.forward([(1.0, 2.0)])
print(y)