English|[中文](README_CN.md)

**This sample provides reference for you to learn the Ascend AI Software Stack and cannot be used for commercial purposes.**

**This README file provides only guidance for running the sample in command line (CLI) mode. For details about how to run the sample in MindStudio, see [Running Speech-to-Text Samples in MindStudio](https://gitee.com/ascend/samples/wikis/Mindstudio%E8%BF%90%E8%A1%8C%E5%9B%BE%E7%89%87%E6%A0%B7%E4%BE%8B?sort_id=3164874).**

## Speech-to-Text Sample
Function: Use the speech conversion model to infer the input speech.   
Input: .bin file converted from a .wav file.   
Output: text converted from the .bin file.    

### Prerequisites
Check whether the following requirements are met. If not, perform operations according to the remarks. If the CANN version is upgraded, check whether the third-party dependencies need to be reinstalled. (The third-party dependencies for 5.0.4 and later versions are different from those for earlier versions.)
| Item| Requirement| Remarks|
|---|---|---|
| CANN version| ≥ 5.0.4| Install the CANN by referring to [Sample Deployment](https://gitee.com/ascend/samples#%E5%AE%89%E8%A3%85) in the *About Ascend Samples Repository*. If the CANN version is earlier than the required version, switch to the **samples** repository specific to the CANN version. See [Release Notes](https://gitee.com/ascend/samples/blob/master/README.md).|
| Hardware| Atlas 200 DK/Atlas 300 ([AI1s](https://support.huaweicloud.com/productdesc-ecs/ecs_01_0047.html#ecs_01_0047__section78423209366)) | Currently, the Atlas 200 DK and Atlas 300 have passed the test. For details about the product description, see [Hardware Platform](https://ascend.huawei.com/en/#/hardware/product). For other products, adaptation may be required.|
| Third-party dependency| ffmpeg + acllite| For details, see [Third-Party Dependency Installation Guide (C++ Sample)](../../../environment).|

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
        # 1. Click Clone or Download in the upper right corner of the samples repository and click Download ZIP.   
        # 2. Upload the .zip package to the home directory of a common user in the development environment, for example, ${HOME}/ascend-samples-master.zip.    
        # 3. In the development environment, run the following commands to unzip the package:    
        cd ${HOME}    
        unzip ascend-samples-master.zip
        ```
2. Obtain the source model required by the application.
    |  **Model** |  **Description** |  **How to Obtain** |
    |---|---|---|
    |  wav2word| Speech-to-text inference model. |  Download the model and weight files by referring to the links in **README.md** in the [ATC_wav2word_tf_AE](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/nlp/wav2word/ATC_wav2word_tf_AE) directory of the ModelZoo repository.|
    ```
    # To facilitate download, the commands for downloading the original model and converting the model are provided here. You can directly copy and run the commands. You can also refer to the above table to download the model from ModelZoo and manually convert it.    
    cd ${HOME}/samples/cplusplus/level2_simple_inference/5_nlp/WAV_to_word/model    
    wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/Wav2word/Wav2word.pb  
    atc --input_shape="the_input:1,1600,200,1" --input_format=NHWC --output=voice --soc_version=Ascend310 --framework=3 --model="./Wav2word.pb"
    ```
### Sample Deployment
Run the following commands to execute the compilation script to start sample compilation:  
```
cd ${HOME}/samples/cplusplus/level2_simple_inference/5_nlp/WAV_to_word/scripts    
bash sample_build.sh
```
### Sample Running

**Note: If the development environment and operating environment are set up on the same server, skip step 1 and go to [step 2](#step_2) directly.**  

1. Run the following commands to upload the **WAV_to_word** directory in the development environment to any directory in the operating environment, for example, **/home/HwHiAiUser**, and log in to the operating environment (host) as the running user (**HwHiAiUser**):
    ```
    # In the following information, xxx.xxx.xxx.xxx is the IP address of the operating environment. The IP address of Atlas 200 DK is 192.168.1.2 when it is connected over the USB port, and that of Atlas 300 (AI1s) is the corresponding public IP address.
    scp -r ${HOME}/samples/cplusplus/level2_simple_inference/5_nlp/WAV_to_word HwHiAiUser@xxx.xxx.xxx.xxx:/home/HwHiAiUser    
    ssh HwHiAiUser@xxx.xxx.xxx.xxx     
    cd ${HOME}/WAV_to_word/scripts
    ```
2. <a name="step_2"></a>Execute the script to run the sample. 
    ```
    bash sample_run.sh
    ```
### Result Viewing
After the running is complete, the inference result is displayed in the CLI of the operating environment.     
![输入图片说明](https://images.gitee.com/uploads/images/2021/1109/100448_942c90d0_5400693.png "屏幕截图.png")

### Common Errors
For details about how to rectify the errors, see [Troubleshooting](https://gitee.com/ascend/samples/wikis/%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98%E5%AE%9A%E4%BD%8D/%E4%BB%8B%E7%BB%8D). If an error is not included in Wiki, submit an issue to the **samples** repository.
