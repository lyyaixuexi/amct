[中文](README_CN.md) | English

**This sample provides reference for you to learn the Ascend AI Software Stack and cannot be used for commercial purposes.**

**This README file provides only guidance for running the sample in command line (CLI) mode. For details about how to run the sample in MindStudio, see [Running Video Samples in MindStudio](https://gitee.com/ascend/samples/wikis/Mindstudio%E8%BF%90%E8%A1%8C%E8%A7%86%E9%A2%91%E6%A0%B7%E4%BE%8B?sort_id=3170138).**

## Sample of Semantic Segmentation Inference on Human Bodies in the Input Video
Function: Uses the semantic segmentation model to perform semantic segmentation inference on human bodies in the input video.   
Input: MP4 video that contains human bodies.   
Output: Inference result displayed on the Presenter Server WebUI.   

### Prerequisites
Check whether the following requirements are met. If not, perform operations according to the remarks. If the CANN version is upgraded, check whether the third-party dependencies need to be reinstalled. (The third-party dependencies for 5.0.4 and later versions are different from those for earlier versions.)

| Item| Requirement| Remarks|
|---|---|---|
| CANN version| >=5.0.4 | Install the CANN by referring to [Installation](/README.md#installation) in the *About Ascend Samples Repository*. If the CANN version is earlier than the required version, switch to the **samples** repository specific to the CANN version. See [Release Notes](/README.md#release-notes).|
| Hardware| Atlas 200 DK/Atlas 300 (ai1s)  | Currently, the Atlas 200 DK and Atlas 300 have passed the test. For details about the product description, see [Hardware Platform](https://ascend.huawei.com/en/#/hardware/product). For other products, adaptation may be required.|
| Third-party dependency| presentagent, opencv, ffmpeg+acllite | For details, see [Third-Party Dependency Installation Guide (C++ Sample)](../../environment/README.md).|

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
    |  human_segmentation| Human body semantic segmentation inference model. It is a TensorFlow-based semantic segmentation model. |  Download the model and weight files by referring to the links in **README.md** in the [ATC_yolov3_caffe_AE](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/human_segmentation/ATC_human_segmentation_tf_AE) directory of the ModelZoo repository.|

    ```
    # To facilitate download, the commands for downloading the original model and converting the model are provided here. You can directly copy and run the commands. You can also refer to the above table to download the model from ModelZoo and manually convert it.    
    
    cd $HOME/samples/cplusplus/contrib/human_segmentation/model    
    wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/human_segmentation/human_segmentation.pb   
    wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/human_segmentation/insert_op.cfg
    atc --input_shape="input_rgb:1,512,512,3" --input_format=NHWC --output=human_segmentation --soc_version=Ascend310 --insert_op_conf=./insert_op.cfg --framework=3 --model=./human_segmentation.pb
    ```


### Sample Deployment
Run the following commands to execute the compilation script to start sample compilation:    
```
cd  $HOME/samples/cplusplus/contrib/human_segmentation/scripts    
bash sample_build.sh
```

### Sample Running
1. Run the following commands to upload the **human_segmentation** directory in the development environment to any directory in the operating environment, for example, **/home/HwHiAiUser**, and log in to the operating environment (host) as the running user (**HwHiAiUser**):     
   ```
   # In the following information, <xxx.xxx.xxx.xxx> is the IP address of the operating environment. The IP address of Atlas 200 DK is 192.168.1.2 when it is connected over the USB port, and that of Atlas 300 (AI1s) is the corresponding public IP address.
   scp -r $HOME/samples/cplusplus/contrib/human_segmentation HwHiAiUser@xxx.xxx.xxx.xxx:/home/HwHiAiUser    
   ssh HwHiAiUser@xxx.xxx.xxx.xxx     
   cd $HOME/samples/cplusplus/contrib/human_segmentation/scripts
   ```

2. <a name="step_2"></a>Execute the script to run the sample.         
   ```
   bash sample_run.sh
   ```

### Result Viewing
1. Open the Presenter Server WebUI.  
   - For Atlas 200 DK:   
     Open the URL that is displayed when Presenter Server is started.  
   - For Atlas 300 AI accelerator card (AI1s cloud inference environment):   
     **The following assumes that the intranet IP address of the Atlas 300 AI accelerator card (AI1s) is 192.168.0.194 and the public IP address is 124.70.8.192.**    
     The message "Please visit http://192.168.0.194:7009 for display server" is displayed when Presenter Server is started.   
     Replace the intranet IP address **192.168.0.194** in the URL with the public IP address **124.70.8.192**. That is, change the URL to http://124.70.8.192:7009.   
     Open the URL in the browser.   
2. Wait for Presenter Agent to transmit data to the server and click **Refresh**. When data arrives, the icon in the **Status** column for the corresponding **Channel** changes to green.   
3. Click a link in the **View Name** column to view the result.   

### Common Errors
For details about how to rectify the errors, see [Troubleshooting](https://gitee.com/ascend/samples/wikis/%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98%E5%AE%9A%E4%BD%8D/%E4%BB%8B%E7%BB%8D). If an error is not included in Wiki, submit an issue to the **samples** repository.
