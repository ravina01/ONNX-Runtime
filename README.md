# ONNX-Runtime
ONNX Runtime

---
![image](https://github.com/user-attachments/assets/9b33393a-6f00-4c03-a695-b53cb6745074)


- deploying on edge devices
-  onnx convrtes ml framework between different ML frameworks
-  pytorch - onnx - onnxruntime
-  to optimize the model in inference environment, its important to have it quantized.

  
-  whats quantized ?
-  if your model contains float32/64 values, -> int32 / 64, will be sacrificng the accuracy, but the predictions would be faster.
-  deploy on edge devices with onnxrutime.
-  

- onnx file format : we can visualize this using netron.app - model graphs 
  
- Model -
- version info
- Metadata
- acyclic computation dataflow graph

- Graph -
- Inputs and outputs
- List of computation nodes
- Graph name

- computation nodes
- operators - activation, batch norm etc
- opertaor paras
- inputs of defined types
- outputs of defined types

Resource - https://github.com/onnx/onnx/blob/main/docs/Operators.md


- ONNX data type
- 
  
![image](https://github.com/user-attachments/assets/9175ee6b-2dc1-4d30-a79d-c8ee42c1ef53)


- ONNX Runtime ->
- High performace inference engine for onnx
- founded by microsoft
- full onnx spec support (v1.2+)
- extensible and modular framework
- 
![image](https://github.com/user-attachments/assets/1c301e86-0f5d-42cc-afd5-89763f993c08)








