中文|[English](README.md)

**本样例为大家学习昇腾软件栈提供参考，非商业目的！**

**本README只提供命令行方式运行样例的指导，如需在Mindstudio下运行样例，请参考[Mindstudio运行视频样例wiki](https://gitee.com/ascend/samples/wikis/Mindstudio%E8%BF%90%E8%A1%8C%E8%A7%86%E9%A2%91%E6%A0%B7%E4%BE%8B?sort_id=3170138)。**

## ascendcamera摄像头样例
功能：使用摄像头拍摄照片或视频。    
样例输入：摄像头(树莓派V1.3版本Camera，暂时只支持到15fps；树莓派V2.1版本Camera，暂时只支持到20fps)    
样例输出：presenter界面展现推理结果，或者数据保存至本地。     

### 前置条件
请检查以下条件要求是否满足，如不满足请按照备注进行相应处理。如果CANN版本升级，请同步检查第三方依赖是否需要重新安装（5.0.4及以上版本第三方依赖和5.0.4以下版本有差异，需要重新安装）。
| 条件 | 要求 | 备注 |
|---|---|---|
| CANN版本 | >=5.0.4 | 请参考CANN样例仓介绍中的[安装步骤](https://gitee.com/ascend/samples#%E5%AE%89%E8%A3%85)完成CANN安装，如果CANN低于要求版本请根据[版本说明](https://gitee.com/ascend/samples/blob/master/README_CN.md#%E7%89%88%E6%9C%AC%E8%AF%B4%E6%98%8E)切换samples仓到对应CANN版本 |
| 硬件要求 | Atlas200DK | 摄像头样例仅在Atlas200D测试及运行，产品说明请参考[硬件平台](https://ascend.huawei.com/zh/#/hardware/product)|
| 第三方依赖 | opencv | 请参考[第三方依赖安装指导（C++样例）](../../../environment)完成对应安装 |

### 样例准备
 可以使用以下两种方式下载源码包，请选择其中一种进行源码准备。   
  - 命令行方式下载（下载时间较长，但步骤简单）。
     ```    
     # 开发环境，非root用户命令行中执行以下命令下载源码仓。    
     cd ${HOME}     
     git clone https://gitee.com/ascend/samples.git
     ```
     **注：如果需要切换到其它tag版本，以v0.5.0为例，可执行以下命令。**
     ```
     git checkout v0.5.0
     ```   
  - 压缩包方式下载（下载时间较短，但步骤稍微复杂）。   
     **注：如果需要下载其它版本代码，请先请根据前置条件说明进行samples仓分支切换。**   
     ``` 
      # 1. samples仓右上角选择 【克隆/下载】 下拉框并选择 【下载ZIP】。    
      # 2. 将ZIP包上传到开发环境中的普通用户家目录中，【例如：${HOME}/ascend-samples-master.zip】。     
      # 3. 开发环境中，执行以下命令，解压zip包。     
      cd ${HOME}    
      unzip ascend-samples-master.zip
      ```
### 样例部署
1. 修改present相关配置文件。    
    将样例目录下**scripts/param.conf**中的 presenter_server_ip、presenter_view_ip 修改为开发环境中可以ping通运行环境的ip地址。   
      1. 开发环境中使用ifconfig查看可用ip。   
      2. 在开发环境中将**scripts/param.conf**中的 presenter_server_ip、presenter_view_ip 修改为该ip地址。   
      ![](https://images.gitee.com/uploads/images/2020/1106/160652_6146f6a4_5395865.gif "icon-note.gif") **说明：**  
      >- 1.开发环境和运行环境分离部署，一般使用配置的虚拟网卡ip，例如192.168.1.223。   
      >- 2.开发环境和运行环境合一部署，一般使用200dk固定ip，例如192.168.1.2。  
  
2. 切换到ascendcamera目录，创建目录用于存放编译文件，例如，本文中，创建的目录为 **build/intermediates/host**。
   ```
   cd $HOME/samples/cplusplus/level1_single_api/5_200dk_peripheral/ascendcamera
   mkdir -p build/intermediates/host
   ```
3. 切换到 **build/intermediates/host** 目录，执行cmake生成编译文件。
   ```
   cd build/intermediates/host   
   make clean   
   cmake ../../../src -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ -DCMAKE_SKIP_RPATH=TRUE
   ```
4. 执行make命令，生成的可执行文件main在 **ascendcamera/out** 目录下。
   ```
   make
   ```
### 样例运行（图片保存至本地）
**注：开发环境与运行环境合一部署，请跳过步骤1，直接执行[步骤2](#step_2)即可。**      
1. 执行以下命令,将开发环境的 **ascendcamera** 目录上传到运行环境中，例如 **/home/HwHiAiUser**，并以HwHiAiUser（运行用户）登录运行环境（Host）。
   ```
   # 【xxx.xxx.xxx.xxx】为运行环境ip，200DK在USB连接时一般为192.168.1.2。
   scp -r $HOME/samples/cplusplus/level1_single_api/5_200dk_peripheral/ascendcamera HwHiAiUser@xxx.xxx.xxx.xxx:/home/HwHiAiUser
   ssh HwHiAiUser@xxx.xxx.xxx.xxx    
   ```
2. <a name="step_2"></a>运行可执行文件。     
    - 如果是开发环境与运行环境合一部署，执行以下命令切换目录。
      ```
      cd $HOME/samples/cplusplus/level1_single_api/5_200dk_peripheral/ascendcamera/out
      ```
    - 如果是开发环境与运行环境分离部署，执行以下命令切换目录。
      ```
      cd $HOME/ascendcamera/out
      ```
    切换目录后，执行以下命令运行样例。
    ```
    ./main -i -c 1 -o ./output/filename.jpg --overwrite
    ```
参数说明：
 -   -i：代表获取jpg格式的图片。       
 -   -c：表示摄像头所在的channel，此参数有“0”和“1”两个选项，“0“对应“Camera1“，“1“对应“Camera2“，如果不填写，默认为“0”。    
 -   -o：表示文件存储位置，此处output为本地已存在的文件夹名称，filename.jpg为保存的图片名称，可用户自定义。    
 -   --overwrite：覆盖已存在的同名文件。   

### 查看结果
运行完成后，会在运行环境的命令行中打印出运行结果,并在将运行结果保存在$HOME/ascendcamera/out/output。    

### 样例运行（视频保存至本地）(这里需要连接camera1)  
**注：开发环境与运行环境合一部署，请跳过步骤1，直接执行[步骤2](#step2_2)即可。**         
1. 执行以下命令,将开发环境的 **ascendcamera** 目录上传到运行环境中，例如 **/home/HwHiAiUser**，并以HwHiAiUser（运行用户）登录运行环境（Host）。
   ```
   # 【xxx.xxx.xxx.xxx】为运行环境ip，200DK在USB连接时一般为192.168.1.2。
   scp -r $HOME/samples/cplusplus/level1_single_api/5_200dk_peripheral/ascendcamera HwHiAiUser@xxx.xxx.xxx.xxx:/home/HwHiAiUser
   ssh HwHiAiUser@xxx.xxx.xxx.xxx    
   ```
2. <a name="step2_2"></a>运行可执行文件。      
    - 如果是开发环境与运行环境合一部署，执行以下命令切换目录。
      ```
      cd $HOME/samples/cplusplus/level1_single_api/5_200dk_peripheral/ascendcamera/out
      ```
    - 如果是开发环境与运行环境分离部署，执行以下命令切换目录。
      ```
      cd $HOME/ascendcamera/out
      ```
    运行之前需要在out文件夹下新建output文件夹
    ```
    cd $HOME/ascendcamera/out
    mkdir output
    ```
    切换目录后，执行以下命令运行样例。
    ```
    ./main
    ```
### 查看结果
运行完成后，会在运行环境的命令行中打印出运行结果,并在将运行结果保存在$HOME/ascendcamera/out/output。

### 样例运行（presenterserver）
 **说明：**  
> - 以下出现的**xxx.xxx.xxx.xxx**为运行环境ip，200DK在USB连接时一般为192.168.1.2。

1. 执行以下命令,将开发环境的 **ascendcamera** 目录上传到运行环境中，例如 **/home/HwHiAiUser**。   
   ```
   # 开发环境与运行环境合一部署，请跳过此步骤！   
   scp -r $HOME/samples/cplusplus/level1_single_api/5_200dk_peripheral/ascendcamera HwHiAiUser@xxx.xxx.xxx.xxx:/home/HwHiAiUser
   ```
2. 启动presenterserver并登录运行环境。     
    1. 开发环境中执行以下命令启动presentserver。      
       ```   
       cd $HOME/samples/cplusplus/level1_single_api/5_200dk_peripheral/ascendcamera   
       bash scripts/run_presenter_server.sh  
       ```      
    2. 执行以下命令登录运行环境。     
       ``` 
       # 开发环境与运行环境合一部署，请跳过此步骤！   
       ssh HwHiAiUser@xxx.xxx.xxx.xxx 
       ```      
3. <a name="step_2"></a>运行可执行文件。       
    - 如果是开发环境与运行环境合一部署，执行以下命令切换目录。 
      ```      
      cd $HOME/samples/cplusplus/level1_single_api/5_200dk_peripheral/ascendcameraout
      ```
    - 如果是开发环境与运行环境分离部署，执行以下命令切换目录。
      ```   
      cd $HOME/ascendcamera/out
      ```
    切换目录后，执行以下命令运行样例。并将ip和xxxx修改为对应的ip和端口号。
    ```
    ./main -v -c 1 -t 60 --fps 20 -w 704 -h 576 -s ip:xxxx/presentername
    ```       
    参数说明：    
    -   -v：代表获取摄像头的视频，用来在Presenter Server端展示。     
    -   -c：表示摄像头所在的channel，此参数有0”和1两个选项，0对应Camera0，1对应Camera1，如果不填写，默认为0。     
    -   -t：表示获取60s的视频文件，如果不指定此参数，则获取视频文件直至程序退出。    
    -   -fps：表示存储视频的帧率，取值范围为1\~20，如果不设置此参数，则默认存储的视频帧率为10fps。     
    -   -w：表示存储视频的宽。    
    -   -h：表示存储视频的高。    
    -   -s：后面的ip值为启动Presenter Server时文件scripts/param.conf中presenter_server_ip的IP地址，**xxxx**为Ascendcamera应用对应的Presenter Server服务器的端口号。     
    -   _presentername_：为在Presenter Server端展示的“View Name“，用户自定义，需要保持唯一，只能为大小写字母、数字、“\_”的组合，位数3\~20。     

### 查看结果
1. 打开presentserver网页界面(打开启动Presenter Server服务时提示的URL即可)。   
2. 等待Presenter Agent传输数据给服务端，单击“Refresh“刷新，当有数据时相应的Channel 的Status变成绿色。   
3. 单击右侧对应的View Name链接，查看结果。    
  
### 常见错误
请参考[常见问题定位](https://gitee.com/ascend/samples/wikis/%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98%E5%AE%9A%E4%BD%8D/%E4%BB%8B%E7%BB%8D)对遇到的错误进行排查。如果wiki中不包含，请在samples仓提issue反馈。