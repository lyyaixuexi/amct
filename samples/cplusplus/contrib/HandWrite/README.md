[中文](README_CN.md) | English

**This sample provides reference for you to learn the Ascend AI Software Stack and cannot be used for commercial purposes.**

**This README file provides only guidance for running the sample in command line (CLI) mode. For details about how to run the sample in MindStudio, see [Running Video Samples in MindStudio](https://gitee.com/ascend/samples/wikis/Mindstudio%E8%BF%90%E8%A1%8C%E8%A7%86%E9%A2%91%E6%A0%B7%E4%BE%8B?sort_id=3170138).**

## Handwritten Chinese Character Recognition
Function: This sample recognizes the Chinese characters captured by the camera and displays the recognition result on the Presenter Server WebUI.   
Input: Raspberry Pi camera.   
Output: Semantic segmentation results displayed on the Presenter Server WebUI.  

### Prerequisites
Check whether the following requirements are met. If not, perform operations according to the remarks. If the CANN version is upgraded, check whether the third-party dependencies need to be reinstalled. (The third-party dependencies for 5.0.4 and later versions are different from those for earlier versions.)

| Item| Requirement| Remarks|
|---|---|---|
| CANN version| >=5.0.4 | Install the CANN by referring to [Installation](/README.md#installation) in the *About Ascend Samples Repository*. If the CANN version is earlier than the required version, switch to the **samples** repository specific to the CANN version. See [Release Notes](/README.md#release-notes
).|
| Hardware| Atlas 200 DK | The camera samples are tested and run only on the Atlas 200 DK. For details about the product description, see the [hardware platform](https://www.hiascend.com/en/hardware/product).|
| Third-party dependency| presentagent,ffmpeg+acllite | For details, see [Third-Party Dependency Installation Guide (C++ Sample)](../../environment/README.md).|

### Sample Preparation
1. Obtain the source package.    
   You can download the source code in either of the following ways:  
    - Command line (The download takes a long time, but the procedure is simple.)
       ```    
       # In the development environment, run the following commands as a non-root user to download the source repository:   
       cd ${HOME}     
       git clone https://gitee.com/ascend/samples.git
       ```
       **Note: To switch to another tag (for example, v0.5.0), run the following command:**
       ```
       git checkout v0.5.0
       ```   
    - Compressed package (The download takes a short time, but the procedure is complex.)  
       **Note: If you want to download the code of another version, switch the branch of the samples repository according to the prerequisites.**  
       ``` 
        # 1. Click "Clone or Download" in the upper right corner of the samples repository and click "Download ZIP".   
        # 2. Upload the ZIP package to the home directory of a common user in the development environment, for example, "${HOME}/ascend-samples-master.zip".    
        # 3. In the development environment, run the following commands to unzip the package:    
        cd ${HOME}    
        unzip ascend-samples-master.zip
        ```

2. Convert the model.     
    |  **Model** |  **Description** |  **How to Obtain** |
    |---|---|---|
    |  resnet18 | Handwritten Chinese character recognition model |  Download the model and weight files by referring to the links in **README.md** in the [ATC_yolov3_caffe_AE](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/resnet18/%20ATC_resnet18_caffe_AE) directory of the ModelZoo repository.|

    ```
    # To facilitate download, the commands for downloading the original model and converting the model are provided here. You can directly copy and run the commands. You can also refer to the above table to download the model from ModelZoo and manually convert it.    
    
    cd  $HOME/samples/cplusplus/contrib/HandWrite/model    
    wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/handwrite/resnet.caffemodel  
    wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/handwrite/resnet.prototxt
    wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/handwrite/insert_op.cfg
    atc --model=./resnet.prototxt --weight=./resnet.caffemodel --framework=0 --output=resnet --soc_version=Ascend310 --insert_op_conf=./insert_op.cfg --input_shape="data:1,3,112,112" --input_format=NCHW
    ```

### Sample Deployment
Run the following commands to execute the compilation script to start sample compilation:    
```
cd $HOME/samples/cplusplus/contrib/HandWrite/scripts    
bash sample_build.sh
```

### Sample Running
**Note: If the development environment and operating environment are set up on the same server, skip step 1 and go to [step 2](#step_2) directly.**      
1. Run the following commands to upload the **HandWrite** directory in the development environment to any directory in the operating environment, for example, **/home/HwHiAiUser**, and log in to the operating environment (host) as the running user (**HwHiAiUser**):    
   ```
   # In the following information, <xxx.xxx.xxx.xxx> is the IP address of the operating environment. The IP address of Atlas 200 DK is 192.168.1.2 when it is connected over the USB port, and that of Atlas 300 (AI1s) is the corresponding public IP address.
   scp -r $HOME/samples/cplusplus/contrib/HandWrite HwHiAiUser@xxx.xxx.xxx.xxx:/home/HwHiAiUser    
   ssh HwHiAiUser@xxx.xxx.xxx.xxx     
   cd $HOME/samples/cplusplus/contrib/HandWrite/scripts
   ```

2. <a name="step_2"></a>Execute the script to run the sample.          
   ```
   bash sample_run.sh
   ```

### Result Viewing
1. Open the Presenter Server WebUI (open the URL displayed when Presenter Server is started).   
2. Wait for Presenter Agent to transmit data to the server and click **Refresh**. When data arrives, the icon in the **Status** column for the corresponding **Channel** changes to green.   
3. Click a link in the **View Name** column to view the result.    

### Common Errors
For details about how to rectify the errors, see [Troubleshooting](https://gitee.com/ascend/samples/wikis/%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98%E5%AE%9A%E4%BD%8D/%E4%BB%8B%E7%BB%8D). If an error is not included in Wiki, submit an issue to the **samples** repository.
