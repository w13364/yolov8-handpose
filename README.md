
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
Follow the Raspberry Pi 4 [guide](https://qengineering.eu/install-opencv-on-raspberry-64-os.html).<br>

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

## Running the app.
You can use **Code::Blocks**.
- Load the project file *.cbp in Code::Blocks.
- Select _Release_, not Debug.
- Compile and run with F9.
- You can alter command line arguments with _Project -> Set programs arguments..._ 

Or use **Cmake**.
```
$ cd *MyDir*
$ mkdir build
$ cd build
$ cmake ..
$ make -j4
```

