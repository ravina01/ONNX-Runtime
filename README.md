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

![image](https://github.com/user-attachments/assets/9175ee6b-2dc1-4d30-a79d-c8ee42c1ef53)


- ONNX Runtime ->
- High performace inference engine for onnx
- founded by microsoft
- full onnx spec support (v1.2+)
- extensible and modular framework
- 
![image](https://github.com/user-attachments/assets/1c301e86-0f5d-42cc-afd5-89763f993c08)


![image](https://github.com/user-attachments/assets/1eea7eb0-310f-46d6-9077-a87fd9c1c78e)



---

## ONNX Runtime on the edge ->
![image](https://github.com/user-attachments/assets/c67c5e3e-408a-41a1-ac73-f691ad236377)

- Runs ONNX models
- High performance implementation of onnx spec
- inference and training

- supports Hardware accelaration with execution providers
 cpu, cuda

- cross platforms - windows, linux

- lang - c++, c


In ONNX Runtime, quantization refers to converting a model's weights and activations from high-precision floating-point numbers (e.g., 32-bit floats) to lower precision formats, typically 8-bit integers. This significantly reduces model size and improves inference speed on hardware accelerators optimized for lower precision computations.

Layer fusion is an optimization technique where multiple consecutive layers in a model are merged into a single operation. This reduces the number of computations and memory operations needed during inference, leading to performance improvements.


ONNX Runtime may perform certain layer fusions during model optimization as part of the quantization process. Common fusion examples include:

Convolution (Conv) + BatchNormalization (BN)
Conv + BN + ReLU (activation)
Conv + ReLU


![image](https://github.com/user-attachments/assets/916aef2f-6feb-40f9-b5bc-be21a7f29f93)

![image](https://github.com/user-attachments/assets/7de67a45-6447-4ee7-ac2e-5702b36e548f)

![image](https://github.com/user-attachments/assets/5be2e6f9-faa4-451e-8baa-f1c4bef22222)

![image](https://github.com/user-attachments/assets/60cd46f4-b9c7-41d6-b4bc-cc5785ec4ced)

![image](https://github.com/user-attachments/assets/6013834a-9884-4a26-acb8-910798e37292)



















