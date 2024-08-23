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





---
### Open neural network exchange

![image](https://github.com/user-attachments/assets/6333bada-cb96-47ac-9fa2-1487276b0a6d)

- it works for specific target deployment task - models trained on any framework
- ML Models resarch to production deployment
- onnx model zoo gives pre-trained models
- pytorch -> export as onnx model.
- now you have onnx format file and now you want to run it as fast as you can on edge/cloud devices, onnx runtime is used, 


![image](https://github.com/user-attachments/assets/d0d30ba3-3315-46f4-93fc-7a7d6e683ba3)

![image](https://github.com/user-attachments/assets/6978af02-95f4-4ec7-a034-b1c32696f62e)

- 14.6 performace gain
- 3.6x performance gain
- 
![image](https://github.com/user-attachments/assets/f5e9db58-0f20-4d8d-8cf8-82ac93fe73aa)

![image](https://github.com/user-attachments/assets/4f46ec56-9746-4897-a745-2a5176985562)



- Performance:
- The right section of the image highlights the performance improvements achieved, likely as a result of using ONNX Runtime for model inference:
- Latency reduced by 43%: The time taken to analyze an image and generate metadata is significantly reduced.
- Throughput increased 1.77x: The system can process 1.77 times more images in the same amount of time, indicating enhanced efficiency.
- API cost reduced by 16%: The cost associated with using the API for image analysis is reduced, possibly due to the efficiency gains from faster processing and lower resource consumption.


- Connection to ONNX Runtime:
These improvements—reduced latency, increased throughput, and lower costs—are typical advantages of using ONNX Runtime. ONNX Runtime is an open-source engine that optimizes the performance of machine learning models, making it faster and more cost-effective to run models in production. By deploying models with ONNX Runtime, organizations can achieve better performance metrics, as depicted in this image.


- Model Representation:
- ONNX Model Format: ONNX (Open Neural Network Exchange) models are stored in a binary format that is serialized using Google Protocol Buffers.
- When you save or load an ONNX model, the data is serialized (or deserialized) in this format, allowing for efficient storage and transfer of the model.
- The .onnx files are essentially serialized protobuf messages.
- This protobuf-based format ensures that the model's structure, including the network architecture, weights, and other metadata, can be efficiently stored and shared across different platforms and frameworks.

- For example, if you have a model that requires specific normalization of inputs before inference, this information can be encapsulated in a protobuf message that travels with the model or is part of the pipeline's configuration.
- When you deserialize a .onnx file, you are effectively loading a serialized representation of the ONNX model, which has been encoded using Google Protocol Buffers (protobuf). The deserialization process converts the binary data back into an in-memory representation of the ONNX model, which is structured according to the ONNX protobuf schema.

#### How Protobuf Helps You Understand Transforms in a .onnx File:


1. Model Structure and Layers:

The deserialized ONNX model will contain a series of nodes (representing layers or operations) and edges (representing the data flow between these operations). Each node in the ONNX graph corresponds to an operation, such as a convolution, ReLU, batch normalization, etc.
Protobuf Schema: The ONNX model structure is defined by the ONNX protobuf schema. Once deserialized, you can traverse the model's graph to inspect each node's operation type (e.g., Conv, BatchNormalization, Relu, etc.), inputs, outputs, and parameters.

2. Input Preprocessing:
 Sometimes, preprocessing steps are embedded in the model or implied by the expected input format. By examining the model's input definitions (including expected dimensions and types), you can infer what transformations might be necessary before feeding data into the model.


**While protobuf itself isn't directly visible during the inspection, it's the mechanism that allows you to convert the .onnx file into a structured, inspectable format. Once deserialized, you can use the ONNX model's structure (thanks to protobuf) to understand what transformations are applied at each stage of the model.**


![image](https://github.com/user-attachments/assets/ef0bed25-bee4-4f80-b664-7d7df6c2eec2)
- NETRON APP
- 
![image](https://github.com/user-attachments/assets/87c549da-9306-470d-a360-b633822dbdd2)
- we can also add custom operation in onnx
- 
![image](https://github.com/user-attachments/assets/bd120bea-24dd-41d6-8f5b-ef8c837ba1af)

![image](https://github.com/user-attachments/assets/65a977f0-8c95-42d0-906b-615cd592d85f)

### How a model runs innside onnx runtime ?

- essentially there are 2 phase of running a model, first create a session with your model, you load the model
- after that - you call run APIs, loading a model is to create an in-memory graph representation of the protobuf,
- basically unpacking the protobuf and you create an efficient graph representation in memory.
- after this we go though model optimization/grpah transformations - quantization and then layer fusion.
- getting rid of all unnceesary layers like drop out etc.
- eliminate bunch of nodes - drop out, slice gets eliminated./ bunch of operators.
- fusions happening - conv + batch norn gets fused.
- multiplication + add will get fused.
- we can add custom fusions trough onnxruntime apis
- once the graph is optimized then we begin to partition the graph into different hardware accelerators.
- flow works as follows -> user of api tells onnx runtime -> that this is the list of hardware accelators, where i would like the onnx model to run on, and this is the preferrerd list.
- onnx runtime goes through it in serial order and tries it to assign - graph + node ro a sepcific acceleartors, what we call as the execution provider in the onnx runtime bar lines. this can also be tensorrt
- once a provide is able to excute a certain graph and node will mark it can move onto next provider.
- at the end of this partitioning process what you really get is - subgraphs which are executed by say tensorRT, or by openVino, rest all gets executed by the cpu.
- Note - hence, onnx runtime gauranteed execution even with limited hardware accelators support, it will always run on cpu
- now comes excution part, we go though all the nodes sequentially, if the model is parallel enough you can enable the parallel execution mode.
- or go though node by node sequentially.


![image](https://github.com/user-attachments/assets/1fd30d67-e8a8-471a-b1d4-5d33038c85eb)

![image](https://github.com/user-attachments/assets/0b71227a-9ad6-4da0-9479-68dfc1b9bf04)

![image](https://github.com/user-attachments/assets/740ada29-167d-4f32-9706-83ba596aaa20)

---

#### Understanding ONNX Runtime and Its Role
- ONNX Runtime is an inference engine developed by Microsoft to run machine learning models in the ONNX (Open Neural Network Exchange) format. It allows you to deploy models across different platforms and hardware, providing performance optimizations and hardware acceleration.

#### ONNX Format and Protobuf
When you export a model to the ONNX format, it is serialized using Protocol Buffers (protobuf). This means that your .onnx file contains a binary representation of your model, including its architecture, parameters (weights), and possibly some metadata.

#### Role of ONNX Runtime
ONNX Runtime is responsible for executing (running) the model described in the .onnx file. It reads the protobuf-serialized model, interprets the operations (layers), and performs inference by running data through the model. Here's how it fits into the overall workflow:

1. Loading and Inference
- Load Model: ONNX Runtime loads the serialized .onnx file.
- Inference: ONNX Runtime takes input data, runs it through the model (executing the operations defined in the ONNX graph), and produces an output. This process is optimized for performance, often using techniques like hardware acceleration (e.g., GPU, TPU) and threading.


2. Optimizing the Model
- You can optimize your model at different stages, either before or after deploying it with ONNX Runtime. Optimizations typically include quantization and layer fusion.

Optimization Stages:

Before Deployment:

- Quantization: This process reduces the model's size and computation requirements by converting the weights and activations from higher precision (e.g., FP32) to lower precision (e.g., INT8). Quantization can be done statically or dynamically.
- Layer Fusion: Layer fusion combines consecutive operations (like Conv + ReLU + BatchNorm) into a single operation to improve performance. This is typically done during the graph optimization process.
- Tools like onnxruntime-tools or onnxruntime-quantization can be used to perform these optimizations before the model is deployed.


During Deployment (Runtime Optimization):

- ONNX Runtime Optimizations: When you deploy the model using ONNX Runtime, it can apply additional optimizations at runtime. These optimizations include graph optimizations (such as node elimination and operator fusion), hardware-specific optimizations (like GPU acceleration), and thread management.
- ONNX Runtime handles these optimizations automatically based on the capabilities of the hardware it’s running on.


3. Quantization Process
Quantization is a crucial step in optimizing your model for deployment on resource-constrained devices like specialized SoCs in surveillance cameras.

Static Quantization: This is done before deployment. It requires calibration data (a small sample of the training data) to determine the scale and zero-point for each layer.

Dynamic Quantization: This can be done during inference, where weights are quantized beforehand, but activations are quantized dynamically during runtime.


4. Layer Fusion
Layer fusion is a technique to combine multiple layers into a single operation to reduce computational overhead and improve performance. This process is often performed as part of graph optimization:

Graph Optimization: ONNX Runtime performs graph-level optimizations, including layer fusion, during the model loading process. These optimizations help to reduce the number of operations the model needs to execute, which in turn speeds up inference.



5. End-to-End Process
Let's summarize the end-to-end process of working with ONNX Runtime, from model training to deployment:

1. Train and Export:

- Train your model using a framework like PyTorch or TensorFlow.
- Export the model to the ONNX format using the framework's export tools.

2. Model Optimization:

- Apply optimizations like quantization and layer fusion using ONNX Runtime or associated tools.
- Example: Quantize the model using onnxruntime-quantization.

3. Load the Model with ONNX Runtime:

- Load the optimized .onnx model using ONNX Runtime in your application.
- ONNX Runtime will apply runtime-specific optimizations based on the hardware.

4. Preprocess Input Data:

- Preprocess input data (e.g., video frames from a camera) to match the model's expected input format.

5. Inference:

- Use ONNX Runtime to perform inference, where it reads input data, processes it through the model, and outputs predictions.

6. Post-process Output:

Interpret the model’s output (e.g., detecting objects in a video frame) and use it in your application.

7. Deployment and Monitoring:

-  Deploy the model on the target hardware (e.g., surveillance camera with specialized SoC).
- Monitor the performance and make adjustments if necessary.


---
**Yes, you can run ONNX models using ONNX Runtime with TensorRT as an execution provider. This setup allows you to take advantage of TensorRT's optimizations while keeping the model in the ONNX format**

