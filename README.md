
------------

## Dependencies.
To run the application, you have to:
- OpenCV 64-bit installed.
- Optional: Code::Blocks. (```$ sudo apt-get install codeblocks```)

### Installing the dependencies.
Start with the usual 
```
$ sudo apt-get update 
$ sudo apt-get upgrade
$ sudo apt-get install cmake wget curl
```
#### OpenCV

#### RKNPU2
```
$ git clone https://github.com/airockchip/rknn-toolkit2.git
```
We only use a few files.
```
rknn-toolkit2-master
│      
└── rknpu2
    │      
    └── runtime
        │       
        └── Linux
            │      
            └── librknn_api
                ├── aarch64
                │   └── librknnrt.so
                └── include
                    ├── rknn_api.h
                    ├── rknn_custom_op.h
                    └── rknn_matmul_api.h

$ cd ~/rknn-toolkit2-master/rknpu2/runtime/Linux/librknn_api/aarch64
$ sudo cp ./librknnrt.so /usr/local/lib
$ cd ~/rknn-toolkit2-master/rknpu2/runtime/Linux/librknn_api/include
$ sudo cp ./rknn_* /usr/local/include
```

Or use **Cmake**.
```
$ cd *MyDir*
$ mkdir build
$ cd build
$ cmake ..
$ make -j4
```

onnx导出必须用rknn官网的导，必须要导出４层。onnx导出至你对应的rknn版本。


模型输出必须保持４层，原因是rknn不支持yolo一层输出，会裁减掉yolo的cfl。

<img width="740" height="477" alt="图片" src="https://github.com/user-attachments/assets/e33b4272-53d8-41ba-b780-4ab88a7e8628" />

cpp代码运行示例，帧率问题可以进一步改进，要做其他事情暂时只demo了。
./YoloV8_NPU rk3568/v8hand.rknn <你自己的图片路径>

<img width="339" height="254" alt="图片" src="https://github.com/user-attachments/assets/232b94dc-f168-489f-8456-ae64cc2709aa" />
