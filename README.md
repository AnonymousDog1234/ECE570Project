# ECE570 Project

## 💻: Enviroment
This project was conducted in a Google Colab enviroment using Python 3.9 (higher version may work as well) and the L4 GPU.

## 🔧: Installation
To get started, download the `ECE570_Project.ipynb` file and upload it to your Google Drive. Next, get the [LCDP dataset](https://drive.google.com/drive/folders/1u5QCUFQBkO7O4qKPeXFLgYHJcrD3hKoA) by selecting the 3 vertical dots and selecting "Make a copy" (make sure you are logged into your Google account).

Run the "Connecting to Google Drive" and "Clone Project and Install Packages" sections. When running this code, it may prompt you to "Restart Session". Restart the session and continue. Run the "Unzip the Dataset" sections to break the dataset up into training, testing, and validation folders. Once the data is unzipped, the "Test" code can be ran. 

**Note: if there are issues with not being able to find testing files, check the path to make sure it is where the test files are in the `src/config/ds/test.yaml` file.

## 👏: Acknowledgements
This project is a reimplemenation of the [Color Shift Estimation-and-Correction for Image Enhancement](https://github.com/yiyulics/CSEC/tree/main). Thank you for your project and help! 
